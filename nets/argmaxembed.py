import torch
class ArgMaxEmbed(torch.autograd.Function): # https://discuss.pytorch.org/t/differentiable-argmax/33020/4

    @staticmethod
    def forward(ctx, input, embedding_matrix):
        idx = torch.argmax(input, 1)
        ctx._input_shape = input.shape
        ctx._input_dtype = input.dtype
        ctx._input_device = input.device
        ctx.save_for_backward(idx)
        return idx, torch.nn.functional.embedding(idx, embedding_matrix)

    @staticmethod
    def backward(ctx, grad_output):
        idx, = ctx.saved_tensors
        grad_input = torch.zeros(ctx._input_shape, device=ctx._input_device, dtype=ctx._input_dtype)
        print(idx.shape, grad_output.sum(1, keepdim=True), grad_input.shape)
        grad_input.scatter_(1, idx[:, None], grad_output.sum(1, keepdim=True))
        return grad_input, None