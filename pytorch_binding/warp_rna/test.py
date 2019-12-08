import torch
import unittest
import numpy as np
import warp_rna._C as core


xs = torch.tensor([], dtype=torch.float32)
ys = torch.tensor([], dtype=torch.int)
xn = torch.tensor([], dtype=torch.int)
yn = torch.tensor([], dtype=torch.int)


class RNALossTest(unittest.TestCase):

    def test_contiguous(self):
        xs = torch.tensor(np.zeros((4, 3, 2, 1)), dtype=torch.float32).transpose(0, 1)
        with self.assertRaisesRegex(RuntimeError, "xs must be contiguous"):
            core.rna_loss(xs, ys, xn, yn)

    def test_device(self):
        with self.assertRaisesRegex(RuntimeError, "xs must be located in the CUDA"):
            core.rna_loss(xs, ys, xn, yn)

    def test_shape(self):
        with self.assertRaisesRegex(RuntimeError, "xs must have 4 dimensions"):
            core.rna_loss(xs.cuda(), ys.cuda(), xn.cuda(), yn.cuda())

    def test_type(self):
        ys = torch.tensor([], dtype=torch.long)
        with self.assertRaisesRegex(RuntimeError, "ys must be a Int tensor"):
            core.rna_loss(xs, ys, xn, yn)

    def test_diagonal(self):

        xs = torch.tensor(
            [[[[0.1, 0.6, 0.1, 0.1, 0.1],
               [0.1, 0.1, 0.6, 0.1, 0.1],
               [0.1, 0.1, 0.2, 0.8, 0.1]],
              [[0.1, 0.6, 0.1, 0.1, 0.1],
               [0.1, 0.1, 0.2, 0.1, 0.1],
               [0.7, 0.1, 0.2, 0.1, 0.1]]]],
            dtype=torch.float32)

        xs = torch.nn.functional.log_softmax(xs, dim=-1)
        ys = torch.tensor([[1, 2]], dtype=torch.int)

        xn = torch.tensor([2], dtype=torch.int)
        yn = torch.tensor([2], dtype=torch.int)

        costs, grads = core.rna_loss(
            xs.cuda(), ys.cuda(),
            xn.cuda(), yn.cuda())

        expected_cost = 2.761683127353663

        np.testing.assert_almost_equal(costs.item(), expected_cost, decimal=6)

        expected_grads = np.array(
            [[[[0.0, -1., 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0]],
              [[0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, -1., 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0]]]])

        np.testing.assert_array_almost_equal(grads.cpu().numpy(), expected_grads)

    def test_blank_align(self):

        xs = torch.tensor([[
            [[0.1, 0.6, 0.1, 0.1, 0.1],
             [0.1, 0.1, 0.6, 0.1, 0.1],
             [0.1, 0.1, 0.2, 0.8, 0.1]],
            [[0.1, 0.6, 0.1, 0.1, 0.1],
             [0.1, 0.1, 0.2, 0.1, 0.1],
             [0.7, 0.1, 0.2, 0.1, 0.1]],
            [[0.3, 0.2, 0.1, 0.1, 0.1],
             [0.1, 0.4, 0.2, 0.1, 0.1],
             [0.2, 0.1, 0.2, 0.3, 0.1]]
        ]], dtype=torch.float32)

        xs = torch.nn.functional.log_softmax(xs, dim=-1)
        ys = torch.tensor([[1, 2]], dtype=torch.int)

        xn = torch.tensor([3], dtype=torch.int)
        yn = torch.tensor([2], dtype=torch.int)

        costs, grads = core.rna_loss(
            xs.cuda(), ys.cuda(),
            xn.cuda(), yn.cuda())

        expected_cost = 3.3550581936582096

        np.testing.assert_almost_equal(costs.item(), expected_cost, decimal=6)

        expected_grads = np.array(
            [[[[-0.29989038, -0.70010962, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0]],
              [[0.0, -0.2998903814074949, 0.0, 0.0, 0.0],
               [-0.3318198750872436, 0.0, -0.36828974350526167, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0]],
              [[0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, -0.63171026, 0.0, 0.0],
               [-0.36828974, 0.0, 0.0, 0.0, 0.0]]]])

        np.testing.assert_array_almost_equal(grads.cpu().numpy(), expected_grads)

    def test_warp(self):

        xs = np.random.RandomState(0).randn(1, 33, 3, 5)

        xs = torch.tensor(xs, dtype=torch.float32)

        xs = torch.nn.functional.log_softmax(xs, dim=-1)
        ys = torch.tensor([[1, 2]], dtype=torch.int)

        xn = torch.tensor([33], dtype=torch.int)
        yn = torch.tensor([2], dtype=torch.int)

        costs, grads = core.rna_loss(
            xs.cuda(), ys.cuda(),
            xn.cuda(), yn.cuda())

        expected_cost = 53.468501840135765

        np.testing.assert_almost_equal(costs.item(), expected_cost, decimal=5)

        expected_grads_sum = -33

        np.testing.assert_array_almost_equal(grads.cpu().numpy().sum(), expected_grads_sum, decimal=3)

    def test_batch(self):

        xs = np.zeros((2, 35, 4, 5))
        xs[0, :33, :3] = np.random.RandomState(0).randn(1, 33, 3, 5)
        xs[1, :35, :4] = np.random.RandomState(0).randn(1, 35, 4, 5)

        xs = torch.tensor(xs, dtype=torch.float32)

        xs = torch.nn.functional.log_softmax(xs, dim=-1)
        ys = torch.tensor([[1, 2, 0], [1, 2, 3]], dtype=torch.int)

        xn = torch.tensor([33, 35], dtype=torch.int)
        yn = torch.tensor([2, 3], dtype=torch.int)

        costs, grads = core.rna_loss(
            xs.cuda(), ys.cuda(),
            xn.cuda(), yn.cuda())

        expected_cost = np.array([53.468501840135765, 51.853917105812265])

        np.testing.assert_array_almost_equal(costs.cpu().numpy(), expected_cost, decimal=5)

        grads_sum = grads.cpu().numpy().reshape((2, -1)).sum(axis=-1)

        expected_grads_sum = np.array([-33, -35])

        np.testing.assert_array_almost_equal(grads_sum, expected_grads_sum, decimal=3)

    def test_lengths(self):

        xs = np.random.RandomState(0).randn(2, 20, 21, 3)
        xs = torch.tensor(xs, dtype=torch.float32)
        xs = torch.nn.functional.log_softmax(xs, dim=-1)

        ys = torch.tensor(np.ones((2, 20)), dtype=torch.int)

        xn = torch.tensor([15, 15], dtype=torch.int)
        yn = torch.tensor([1, 1], dtype=torch.int)

        costs, grads = core.rna_loss(
            xs.cuda(), ys.cuda(),
            xn.cuda(), yn.cuda())

        expected_cost = np.array([16.839792588945365, 17.152040043873])

        np.testing.assert_array_almost_equal(costs.cpu().numpy(), expected_cost, decimal=5)

        grads_sum = grads.cpu().numpy().reshape((2, -1)).sum(axis=-1)

        expected_grads_sum = np.array([-15, -15])

        np.testing.assert_array_almost_equal(grads_sum, expected_grads_sum, decimal=3)

    def test_one_to_empty(self):

        xs = torch.tensor([[[[0.1, 0.6, 0.1, 0.1, 0.1]]]], dtype=torch.float32)

        xs = torch.nn.functional.log_softmax(xs, dim=-1)
        ys = torch.tensor([[]], dtype=torch.int)

        xn = torch.tensor([1], dtype=torch.int)
        yn = torch.tensor([0], dtype=torch.int)

        costs, grads = core.rna_loss(
            xs.cuda(), ys.cuda(),
            xn.cuda(), yn.cuda())

        expected_cost = 1.7314291957733714

        np.testing.assert_almost_equal(costs.item(), expected_cost, decimal=6)

        expected_grads = np.array([[[[-1., 0.0, 0.0, 0.0, 0.0]]]])

        np.testing.assert_array_almost_equal(grads.cpu().numpy(), expected_grads)


if __name__ == "__main__":
    unittest.main()
