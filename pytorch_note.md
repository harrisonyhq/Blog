# torch.distributed

开发过程中需要实现List[tensor]在不同tp，甚至跨机的广播，尝试了如下方法，进行记录：

## vllm.distributed.parallel_state:
在vllm源码中，GroupCoordinator类对torch原生distributed接口进行了二次封装：

```python

def broadcast(self, input_: torch.Tensor, src: int = 0):
    """Broadcast the input tensor.
    NOTE: `src` is the local rank of the source rank.
    """
    assert src < self.world_size, f"Invalid src rank ({src})"

    # Bypass the function if we are using only 1 GPU.
    if self.world_size == 1:
        return input_
    # Broadcast.
    torch.distributed.broadcast(
        input_, src=self.ranks[src], group=self.device_group
    )
    return input_

```
这里的broadcast本质上就是封装了torch.distributed.broadcast，只不过group默认为vllm GroupCoordinator实例初始化的group，例如get_tp_group()得到的tp组；

## torch.distributed._broadcast_coalesced
用法：

```python
torch.distributed._broadcast_coalesced(
    process_group: ProcessGroup,
    tensors: list[Tensor],
    buffer_size: int,
    src: int,
    )

```

找到其具体的实现在：

```cpp
// Broadcast many tensors to all processes in the process group.
void broadcast_coalesced(
    c10::intrusive_ptr<c10d::ProcessGroup> process_group,
    at::TensorList tensors,
    size_t buffer_size,
    int rank) {
  // Coalesce tensors into buckets taking into account the maximum buffer size.
  // This routine is multi-device aware, so the tensors can be split across
  // multiple devices and can contain a mix of CPU and CUDA tensors.
  // 首先计算出桶
  const auto buckets =
      compute_bucket_assignment_by_size(tensors.vec(), {buffer_size});

  // Returns tensor at specified index in input tensor list.
  const auto lookup = [&tensors](size_t index) { return tensors[index]; };

  // We maintain a maximum of 2 in flight broadcast operations to avoid
  // allocating too much memory (in case the specified tensors are very large).
  std::deque<BroadcastWork> in_flight; // 建立一个广播work列表
  constexpr auto max_in_flight = 2;
  for (const auto& bucket : buckets) { // 遍历桶
    if (in_flight.size() >= max_in_flight) { // 由注释可以知道，广播维度是2，这样避免内存占用过大
      in_flight.front().finish(); // 广播变量
      in_flight.pop_front();
    }

    in_flight.emplace_back(process_group, c10::fmap(bucket, lookup), rank);
  }

  while (!in_flight.empty()) {
    in_flight.front().finish();
    in_flight.pop_front();
  }
}
```

由C++代码可知，该方法首先遍历所有的tensor，将其按照大小和不同设备等分成桶中的不同的组，组成buffer size大小的buffer，再遍历桶，以最大2个的广播列表将其分组的广播出去。该方法的优点是灵活性，支持tensor list的广播，并且不要求形状，设备一致；但是缺点也很明显，遍历桶，组成buffer，再逐个广播会带来一定的开销。

## 参考
https://www.cnblogs.com/rossiXYZ/p/15584032.html