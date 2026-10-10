import unittest
import torch
from common import Refiner, initial_ref


class TestConditional(unittest.TestCase):
    def test_feather_covers_edges(self):
        import numpy as np
        from infer import mask_at,positions,render
        for H,W in [(512,512),(513,517),(3072,4096)]:
            weights=np.zeros((H,W,1),np.float32)
            for y in positions(H):
                for x in positions(W):weights[y:y+512,x:x+512]+=mask_at(y,x,H,W)
            self.assertGreater(float(weights.min()),0)
        class Identity:
            def infer(self,x,prompt=''):return x
        img=np.full((513,517,3),255,np.uint8)
        self.assertTrue(np.array_equal(render(Identity(),img),img))
    def test_zero_new_channels_equal(self):
        torch.manual_seed(42)
        old=initial_ref(False);new=initial_ref(True)
        x=torch.randn(1,3,32,32);lq=torch.randn_like(x)
        with torch.no_grad():
            self.assertTrue(torch.allclose(old(x,lq),new(x,lq),atol=1e-6,rtol=0))
            self.assertTrue(torch.equal(new.net[0].weight[:,3:],torch.zeros_like(new.net[0].weight[:,3:])))
    def test_extra_input_has_gradient(self):
        new=initial_ref(True)
        x=torch.randn(1,3,32,32);lq=torch.randn_like(x)
        new(x,lq).mean().backward()
        self.assertGreater(new.net[0].weight.grad[:,3:].abs().sum().item(),0)
    def test_size_and_parameter_count(self):
        new=Refiner(True)
        self.assertEqual(sum(p.numel() for p in new.parameters()),5955)
        self.assertEqual(new(torch.zeros(1,3,32,32),torch.zeros(1,3,32,32)).shape,(1,3,32,32))

if __name__=='__main__':unittest.main(verbosity=2)
