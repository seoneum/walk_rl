#pragma once
#include <torch/torch.h>

class EliteBuffer {
public:
  EliteBuffer(int capacity, int obs_dim, int act_dim, torch::Device dev)
      : capacity_(capacity), obs_dim_(obs_dim), act_dim_(act_dim),
        device_(dev) {
    obs_ = torch::empty({capacity_, obs_dim_}, torch::kFloat32).to(device_);
    act_ = torch::empty({capacity_, act_dim_}, torch::kFloat32).to(device_);
    rew_ = torch::empty({capacity_}, torch::kFloat32).to(device_);
  }

  void add_batch(const torch::Tensor &obs, const torch::Tensor &act,
                 const torch::Tensor &rew) {
    auto K = obs.size(0);
    if (K <= 0)
      return;
    auto o = obs.to(device_).to(torch::kFloat32).contiguous();
    auto a = act.to(device_).to(torch::kFloat32).contiguous();
    auto r = rew.to(device_).to(torch::kFloat32).contiguous();

    int64_t pos = head_;
    int64_t room = capacity_ - head_;
    if (K <= room) {
      obs_.narrow(0, pos, K).copy_(o);
      act_.narrow(0, pos, K).copy_(a);
      rew_.narrow(0, pos, K).copy_(r);
    } else { // wrap-around
      obs_.narrow(0, pos, room).copy_(o.narrow(0, 0, room));
      act_.narrow(0, pos, room).copy_(a.narrow(0, 0, room));
      rew_.narrow(0, pos, room).copy_(r.narrow(0, 0, room));
      int64_t rest = K - room;
      obs_.narrow(0, 0, rest).copy_(o.narrow(0, room, rest));
      act_.narrow(0, 0, rest).copy_(a.narrow(0, room, rest));
      rew_.narrow(0, 0, rest).copy_(r.narrow(0, room, rest));
    }
    head_ = (head_ + K) % capacity_;
    size_ = std::min<int64_t>(capacity_, size_ + K);
  }

  std::pair<torch::Tensor, torch::Tensor> sample(int batch) {
    if (size_ <= 0)
      return {torch::Tensor(), torch::Tensor()};
    int B = std::min<int>(batch, (int)size_);
    auto idx = torch::randint(
        0, size_, {B},
        torch::TensorOptions().dtype(torch::kLong).device(device_));
    auto o = obs_.index_select(0, idx);
    auto a = act_.index_select(0, idx);
    return {o, a};
  }

  int64_t size() const { return size_; }

private:
  int64_t capacity_ = 0, size_ = 0, head_ = 0;
  int obs_dim_ = 0, act_dim_ = 0;
  torch::Device device_;
  torch::Tensor obs_, act_, rew_;
};
