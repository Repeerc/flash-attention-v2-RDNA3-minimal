import math
import torch


torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

def pad_to_multiple(tensor, multiple, dim=-1, val = 0):
    length = tensor.size(dim)
    remainder = length % multiple
    if remainder == 0:
        return tensor
    padding_length = multiple - remainder
    padding_shape = list(tensor.shape)
    padding_shape[dim] = padding_length
    padding_tensor = torch.zeros(padding_shape, device=tensor.device, dtype=tensor.dtype) + val
    return torch.cat([tensor, padding_tensor], dim=dim)

class FlashAttentionFunction(torch.autograd.Function):

    @staticmethod
    @torch.no_grad()
    def forward(ctx, q, k, v, mask=None, causal=False, Br=64, Bc=256):

        B = q.shape[0]
        H = q.shape[1]
        N = q.shape[2]
        D = q.shape[3]
        Nkv = k.shape[2]
        scale = D**-0.5

        
        q = pad_to_multiple(q, Br, 2, -100)
        k = pad_to_multiple(k, Bc, 2, -100)
        v = pad_to_multiple(v, Bc, 2)
        
        o = torch.zeros_like(q)
        

        q_frags = torch.split(q, Br, -2)
        k_frags = torch.split(k, Bc, -2)
        v_frags = torch.split(v, Bc, -2)
        o_frags = torch.split(o, Br, -2)

        L = torch.zeros((B, H, q.shape[2]), dtype=torch.float32, device=q.device)

        L_frags = torch.split(L, Br, -1)

        Tr = len(q_frags)
        Tc = len(k_frags)

        for Tr_i in range(Tr):
            row_max_old = (
                torch.zeros((B, H, Br), dtype=q.dtype, device=q.device) - torch.inf
            )
            l_i = torch.zeros((B, H, Br), dtype=q.dtype, device=q.device)
            Oi = torch.zeros((B, H, Br, D), dtype=q.dtype, device=q.device)
            for Tc_j in range(Tc):
                Sij = torch.einsum(
                    "... r D, ... c D -> ... r c", scale * q_frags[Tr_i], k_frags[Tc_j]
                )
                
                if causal:
                    ele_y = Tr_i * Br
                    ele_x = Tc_j * Bc
                    if ele_y < ele_x + Bc - 1:
                        causal_mask = torch.ones((Br, Bc),dtype=torch.bool, device=q.device).triu(ele_y - ele_x + 1)
                        Sij.masked_fill_(causal_mask, -65500.0)
                        
                row_max_new = torch.max(Sij, -1).values
                row_max_new = torch.max(row_max_new, row_max_old)
                Sij = torch.exp(Sij - row_max_new.unsqueeze(-1))
                rowsum = torch.sum(Sij, -1)
                rowmax_diff = row_max_old - row_max_new
                l_i = l_i * torch.exp(rowmax_diff) + rowsum
                Oi *= torch.exp(rowmax_diff).unsqueeze(-1)
                Oi += torch.einsum("... r c, ... c d -> ... r d", Sij, v_frags[Tc_j])
                row_max_old = row_max_new

            Oi = Oi / l_i.unsqueeze(-1)
            o_frags[Tr_i][:] = Oi

            l_i = row_max_old + torch.log(l_i)
            L_frags[Tr_i][:] = l_i

        ctx.args = (causal, scale, mask, Br, Bc, N, Nkv)
        ctx.save_for_backward(q, k, v, o, L)

        return o[:,:,:N,:]

    @staticmethod
    @torch.no_grad()
    def backward(ctx, do):
        causal, scale, mask, Br, Bc, N, Nkv = ctx.args
        q, k, v, o, L = ctx.saved_tensors
        
        #Br = 64
        #Bc = 512
        do = pad_to_multiple(do, Br, 2)
        
        k[:,:,Nkv:,:] = 0

        dQ = torch.zeros_like(q)
        dK = torch.zeros_like(k)
        dV = torch.zeros_like(v)
        

        q_frags = torch.split(q, Br, -2)
        k_frags = torch.split(k, Bc, -2)
        v_frags = torch.split(v, Bc, -2)
        o_frags = torch.split(o, Br, -2)
        dO_frags = torch.split(do, Br, -2)
        dQ_frags = torch.split(dQ, Br, -2)
        dK_frags = torch.split(dK, Bc, -2)
        dV_frags = torch.split(dV, Bc, -2)

        L_frags = torch.split(L, Br, -1)

        Tr = len(q_frags)
        Tc = len(k_frags)
        
        gard_scale = 0.5

        for Tc_j in range(Tc):
            Kj = k_frags[Tc_j]
            Vj = v_frags[Tc_j]
            dKj = dK_frags[Tc_j]
            dVj = dV_frags[Tc_j]
            for Tr_i in range(Tr):
                Qi = q_frags[Tr_i]
                Oi = o_frags[Tr_i]
                dOi = dO_frags[Tr_i]
                dQi = dQ_frags[Tr_i]
                Li = L_frags[Tr_i]
                
                Sij = scale * torch.einsum("... r d, ... c d -> ... r c", Qi, Kj)
                
                if causal:
                    ele_y = Tr_i * Br
                    ele_x = Tc_j * Bc
                    if ele_y < ele_x + Bc - 1:
                        causal_mask = torch.ones((Br, Bc),dtype=torch.bool, device=q.device).triu(ele_y - ele_x + 1)
                        Sij.masked_fill_(causal_mask, -65500.0)
                
                Pij = torch.exp(Sij - Li.unsqueeze(-1)).to(q.dtype)
                dVj += torch.einsum("... r c, ... r d -> ... c d", Pij, dOi) * gard_scale #* 0.01
                dPi = torch.einsum("... r d, ... c d -> ... r c", dOi, Vj)
                Di = torch.sum(dOi * Oi, -1)
                dSij = scale * Pij * (dPi - Di.unsqueeze(-1))
                dQi += torch.einsum("... r c, ... c d -> ... r d", dSij, Kj) * gard_scale #* 0.01
                dKj += torch.einsum("... r c, ... r d -> ... c d", dSij, Qi) * gard_scale #* 0.01
        return dQ[:,:,:N,:], dK[:,:,:Nkv,:], dV[:,:,:Nkv,:], None, None


# (B, H, NQ, D) = 1, 20, 576, 64
# NKV = 227

(B, H, NQ, D) = 1, 20, 2048, 64
NKV = 2048

dtype = torch.float16

if __name__ == "__main__":
    q = torch.rand((B, H, NQ, D), dtype=dtype, device="cuda")  #  * 3.2
    k = torch.rand((B, H, NKV, D), dtype=dtype, device="cuda") #  * 75
    v = torch.rand((B, H, NKV, D), dtype=dtype, device="cuda") #  * 15

    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    q2 = q.clone().detach().requires_grad_(True)
    k2 = k.clone().detach().requires_grad_(True)
    v2 = v.clone().detach().requires_grad_(True)

    fttn = FlashAttentionFunction()
    o1 = fttn.apply(q, k, v, None, True)

    o2 = torch.nn.functional.scaled_dot_product_attention(q2, k2, v2, is_causal=True)

    maxdiff = (o2 - o1).abs().max().item()

    print(o1.cpu()[0, 0, :, :])
    print(o2.cpu()[0, 0, :, :])


    dO = torch.ones_like(q) * 2

    o1.backward(dO)
    o2.backward(dO)

    dQ1 = q.grad.clone().detach()
    dK1 = k.grad.clone().detach()
    dV1 = v.grad.clone().detach()

    dQ2 = q2.grad.clone().detach()
    dK2 = k2.grad.clone().detach()
    dV2 = v2.grad.clone().detach()

    
    print("FTTN dQ",dQ1.cpu()[0,-1,:,:] * 1e4)
    print('PT dQ',dQ2.cpu()[0,-1,:,:] * 1e4)
    
    print("FTTN dK",dK1.cpu()[0,-1,:,:] )
    print('PT dK',dK2.cpu()[0,-1,:,:] )
    
    print("FTTN dV",dV1.cpu()[0,-1,:,:] )
    print('PT dV',dV2.cpu()[0,-1,:,:] )
    
    print("fwd diff:", maxdiff)
    print(f"dQ diff:{(dQ1 - dQ2).abs().max().item()}")
    print(f"dK diff:{(dK1 - dK2).abs().max().item()}")
    print(f"dV diff:{(dV1 - dV2).abs().max().item()}")
