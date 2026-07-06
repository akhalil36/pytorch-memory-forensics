import struct
import json
import logging
from typing import List, Iterable, Tuple, Optional

from volatility3.framework import interfaces, renderers
from volatility3.framework.configuration import requirements
from volatility3.framework.renderers import format_hints
from volatility3.plugins.linux import pslist
from volatility3.framework.layers import scanners

vollog = logging.getLogger(__name__)

class PyTorchExtractor(interfaces.plugins.PluginInterface):
    """
    extracts the model architecture, layer names, and weights. 
    also retrieves the forward method's execution flow information.
    """
    _required_framework_version = (2, 0, 0)

    @classmethod
    def get_requirements(cls):
        return [
            requirements.ModuleRequirement(
                name="kernel",
                description="Linux kernel",
                architectures=["Intel64"],
            ),
            requirements.IntRequirement(
                name="pid",
                description="PID of the PyTorch inference process",
                optional=False,
            ),
            requirements.StringRequirement(
                name="output",
                description="Output file path for weights JSON",
                default="/tmp/pytorch_extracted.json",
                optional=True,
            ),
        ]
    
    def run(self):
        pid = self.config["pid"]
        return renderers.TreeGrid(
            [
                ("Layer", str),
                ("Type", str),
                ("Shape", str),
                ("Strides", str),
                ("DataPtr", format_hints.Hex),
                ("NumElements", int),
            ],
            self._generator(pid),
        )
    
    def _generator(self, pid: int):
        """outputs one row per tensor that has been found"""
        kernel = self.context.modules[self.config["kernel"]]
        proc = self._find_process(kernel, pid)

        layer = self.context.layers[proc.add_process_layer()]
        heap_ranges = self._get_heap_ranges(proc)

        # put the addresses of the layers I found using GDB & volshell here
        tensors_gdb_addresses = {
            "conv2d_1.weight": (0x5555583dc970, 0x5555583e2500),
            "conv2d_1.bias": (0x5555583d3270, 0x55555841fcc0),

            "conv2d_2.weight": (0x5555583cd820, 0x55555842af40),
            "conv2d_2.bias": (0x5555583d1660, 0x5555583e5c40),

            "conv2d_3.weight": (0x5555583d57d0, 0x55555843cf80),
            "conv2d_3.bias": (0x5555583c9a30, 0x5555583ec9c0),

            "linear_1.weight": (0x5555583f33c0, 0x7fffe23ff040),
            "linear_1.bias": (0x5555583f11d0, 0x5555583e7d00),

            "linear_2.weight": (0x5555583d0480, 0x555558484fc0),
            "linear_2.bias": (0x5555583ea620, 0x5555583cc480),
        }

        all_weights = {}

        for layer_name, (tensor_impl_addr, override_ptr) in tensors_gdb_addresses.items():
            result = self._read_tensor_impl(layer, tensor_impl_addr)
            if not result:
                continue

            shape, strides, storage_ptr, n_elems = result
            data_ptr = None
            if override_ptr:
                data_ptr = override_ptr
            else:
                data_ptr = self._read_data_ptr(layer, storage_ptr)

            weights = []
            if data_ptr:
                weights = self._extract_weights(layer, data_ptr, n_elems)
                all_weights[layer_name] = {
                    "shape": shape,
                    "strides": strides,
                    "weights": weights,
                }
                self._print_weights_preview(layer, layer_name, data_ptr, 10)
            
            layer_type = "Linear"
            if "conv" in layer_name:
                layer_type = "Conv2d"

            yield (0, (
                layer_name,
                layer_type,
                str(shape),
                str(strides),
                format_hints.Hex(data_ptr or 0),
                n_elems,
            ))

        architecture = self._extract_architecture(layer, heap_ranges)

        # put the output in the json
        out_path = self.config["output"]

        # retrieving the code from the forward pass
        vollog.info("Scanning TorchScript forward method code -")
        forward_ops = self._extract_forward_code(layer)

        print("\n" + "--*--"*60)
        print("CODE FROM THE FORWARD PASS")
        print("--*--"*60)
        print("TorchScript TreeGrid from print_graph.py script:")
        print("--*--"*60)

        # I got this using a script print_graph.py which is included in the blog
        # helpful to include the results here for reference

        print("""graph(%self.1 : __torch__.model.CNN,
      %x.1 : Tensor):
  %features.1 = prim::GetAttr[name="features"](%self.1)
  %x0.1 = prim::CallMethod[name="forward"](%features.1, %x.1)
  %classifier.1 = prim::GetAttr[name="classifier"](%self.1)
  %11 = prim::CallMethod[name="forward"](%classifier.1, %x0.1)
  return (%11)""")
        
        print("\nOperator strings found in memory image:")
        print("--*--"*60)
        print(f"{'Operator':<30} {'Address'}")
        print("--*--"*60)

        for op_name, op_addr in forward_ops:
            print(f"{op_name:<30} {op_addr}")

        print("="*60 + "\n")

        for op_name, op_addr in forward_ops:
            yield (0, (
                "forward_code",
                "TorchScript Operator",
                op_name,
                "",
                # ensures that Volatility outputs the value as a hex number
                format_hints.Hex(int(op_addr, 16)),
                0,
            ))

        # writing the results to the gjson
        output = {
            "architecture": architecture,
            "weights": all_weights,
            "forward_code": [{"operator": op, "address": addr} for op, addr in forward_ops],
        }

        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

        vollog.info(f"Wrote extracted model to {out_path}")

    def _find_process(self, kernel, pid: int):
        """finds the process matching my PID"""
        tasks = pslist.PsList.list_tasks(self.context, kernel.name, filter_func=pslist.PsList.create_pid_filter([pid]))
        for task in tasks:
            return task
        return None                       
        
    
    def _build_proc_layer(self, kernel, proc, layer_name: str):
        """sets up the physical to virtual address translation"""
        return proc.add_process_layer()
    
    def _get_heap_ranges(self, proc) -> List[Tuple[int, int]]:
        return [(0x555555585000, 0x5555584a5000)]
    
    def _read_tensor_impl(self, layer, addr: int):
        """
        reads the TensorImpl info that I already found using GDB
        """
        n_dim = struct.unpack("<Q", layer.read(addr + 0x40, 8))[0]
        if not (0 < n_dim <= 8):
            return None

        sizes = []
        for i in range(n_dim):
            val = struct.unpack("<q", layer.read(addr + 0x48 + i * 8, 8))[0]
            sizes.append(val)

        strides = []
        base = addr + 0x48 + n_dim * 8
        for i in range(n_dim):
            val = struct.unpack("<q", layer.read(base + i * 8, 8))[0]
            strides.append(val)

        n_elems = 1
        for s in sizes:
            n_elems *= s

        storage_ptr = struct.unpack("<Q", layer.read(addr + 0x08, 8))[0]

        return sizes, strides, storage_ptr, n_elems
    
    def _read_data_ptr(self, layer, storage_ptr: int) -> Optional[int]:
        """
        get the data pointer using the addresses I documented in the blog
        which I found using GDB.
        """
        impl_ptr = struct.unpack("<Q", layer.read(storage_ptr, 8))[0]
        if not impl_ptr:
            return None

        data_ptr = struct.unpack("<Q", layer.read(impl_ptr + 0x08, 8))[0]
        if not data_ptr:
            return None
        return data_ptr
    
    def _extract_weights(self, layer, data_ptr: int, n_elems: int) -> List[float]:
        """ get the weights """
        weights = []
        for i in range(n_elems):
            value = struct.unpack("<f", layer.read(data_ptr + i * 4, 4))[0]
            weights.append(value)
        return weights
        
    
    def _scan_heap_for_tensors(self, layer, heap_ranges) -> dict:
        """this tries to interpret every 8 bytes of the heap as a TensorImpl obj"""
        found = {}
        for start, end in heap_ranges:
            for addr in range(start, end - 0x80, 8):
                result = self._read_tensor_impl(layer, addr)
                if result:
                    sizes, _, storage_ptr, n_elems = result
                    # the sizes must be reasonable for a CNN
                    if (all(0 < s < 100000 for s in sizes) and
                        0 < n_elems < 10_000_000 and
                        storage_ptr > 0x555555000000):
                        found[addr] = result

        return found
    
    def _extract_architecture(self, layer, heap_ranges) -> List[dict]:
        """
        extracts ClassType objects that I have already found using GDB
        """
        architecture = []
        known_layer_names = [b"conv2d", b"Linear", b"BatchNorm", b"ReLU", b"MaxPool", b"Flatten"]

        for start, end in heap_ranges:
            for name_bytes in known_layer_names:
                hits = layer.scan(
                    self.context,
                    scanners.BytesScanner(name_bytes),
                    sections=[(start, end - start)],
                )
                for hit in hits:
                    length = struct.unpack("<Q", layer.read(hit - 8, 8))[0]
                    if length == len(name_bytes):
                        name = name_bytes.decode("ascii")
                        vollog.info(f"Found layer type '{name}' @ {hex(hit)}")
                        architecture.append({
                            "address": hex(hit),
                            "layer_type": name,
                            "attributes": self._read_layer_attributes(layer, hit + len(name_bytes)),
                        })
        return architecture
    
    def _read_layer_attributes(self, layer, addr: int) -> dict:
        """
        finds strings such as "padding", "dilation", and "padding_mode"
        """
        attrs = {}
        known_attrs = [b"stride", b"padding", b"dilation", b"padding_mode",
                       b"in_features", b"out_features", b"bias"]
        try:
            region = layer.read(addr, 256)
            for attr in known_attrs:
                if attr in region:
                    attrs[attr.decode()] = True
        except Exception:
            pass
        return attrs
    
    def _extract_forward_code(self, layer):
        """
        looks in the heap for info located in the forward graph found
        by hte print_graph.py script I wrote
        """
        heap_start = 0x555555585000
        heap_end = 0x5555584a5000

        targets = [
            b"prim::GetAttr",
            b"prim::CallMethod",
            b"aten::conv2d",
            b"aten::relu_",
            b"aten::max_pool2d",
            b"aten::linear",
            b"/app/model.py",
            b"__torch__.model.CNN",
        ]

        results = []
        for target in targets:
            try:
                hits = layer.scan(
                    self.context,
                    scanners.BytesScanner(target),
                    sections=[(heap_start, heap_end - heap_start)]
                )
                for hit in hits:
                    results.append((target.decode(), hex(hit)))
                    break
            except Exception as e:
                vollog.debug(f"Scan failed for {target}: {e}")

        return results
    
    def _print_weights_preview(self, layer, name, data_ptr, n_floats):
        """prints some of the float weights found"""
        print(f"\n{'=*'*60}")
        print(f"Weights preview: {name}")
        print(f"DataPtr: {hex(data_ptr)}")
        print(f"{'--*--'*60}")
        
        floats = []
        for i in range(n_floats):
            
            value = struct.unpack("<f", layer.read(data_ptr + i * 4, 4))[0]
            floats.append(value)

    
        for i in range(0, len(floats), 2):
            addr = data_ptr + i * 4
            vals = "\t".join(f"{v:.19e}" for v in floats[i:i+2])
            print(f"0x{addr:016x}:\t{vals}")
        
        print(f"{'--*--'*60}")
        print(f"Total floats read: {len(floats)}")