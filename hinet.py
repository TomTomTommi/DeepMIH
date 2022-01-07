from model import *
from invblock import INV_block_addition
from invblock import INV_block_affine

class Hinet_stage1(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self):
        super(Hinet_stage1, self).__init__()

        self.inv1 = INV_block_affine(imp_map=False)
        self.inv2 = INV_block_affine(imp_map=False)
        self.inv3 = INV_block_affine(imp_map=False)
        self.inv4 = INV_block_affine(imp_map=False)
        self.inv5 = INV_block_affine(imp_map=False)
        self.inv6 = INV_block_affine(imp_map=False)
        self.inv7 = INV_block_affine(imp_map=False)
        self.inv8 = INV_block_affine(imp_map=False)

        self.inv9 = INV_block_affine(imp_map=False)
        self.inv10 = INV_block_affine(imp_map=False)
        self.inv11 = INV_block_affine(imp_map=False)
        self.inv12 = INV_block_affine(imp_map=False)
        self.inv13 = INV_block_affine(imp_map=False)
        self.inv14 = INV_block_affine(imp_map=False)
        self.inv15 = INV_block_affine(imp_map=False)
        self.inv16 = INV_block_affine(imp_map=False)

    def forward(self, x, rev=False):

        if not rev:
            out = self.inv1(x)
            out = self.inv2(out)
            out = self.inv3(out)
            out = self.inv4(out)
            out = self.inv5(out)
            out = self.inv6(out)
            out = self.inv7(out)
            out = self.inv8(out)

            out = self.inv9(out)
            out = self.inv10(out)
            out = self.inv11(out)
            out = self.inv12(out)
            out = self.inv13(out)
            out = self.inv14(out)
            out = self.inv15(out)
            out = self.inv16(out)

        else:
            out = self.inv16(x, rev=True)
            out = self.inv15(out, rev=True)
            out = self.inv14(out, rev=True)
            out = self.inv13(out, rev=True)
            out = self.inv12(out, rev=True)
            out = self.inv11(out, rev=True)
            out = self.inv10(out, rev=True)
            out = self.inv9(out, rev=True)

            out = self.inv8(out, rev=True)
            out = self.inv7(out, rev=True)
            out = self.inv6(out, rev=True)
            out = self.inv5(out, rev=True)
            out = self.inv4(out, rev=True)
            out = self.inv3(out, rev=True)
            out = self.inv2(out, rev=True)
            out = self.inv1(out, rev=True)

        return out


class Hinet_stage2(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self):
        super(Hinet_stage2, self).__init__()

        self.inv1 = INV_block_affine()
        self.inv2 = INV_block_affine()
        self.inv3 = INV_block_affine()
        self.inv4 = INV_block_affine()
        self.inv5 = INV_block_affine()
        self.inv6 = INV_block_affine()
        self.inv7 = INV_block_affine()
        self.inv8 = INV_block_affine()

        self.inv9 = INV_block_affine()
        self.inv10 = INV_block_affine()
        self.inv11 = INV_block_affine()
        self.inv12 = INV_block_affine()
        self.inv13 = INV_block_affine()
        self.inv14 = INV_block_affine()
        self.inv15 = INV_block_affine()
        self.inv16 = INV_block_affine()

    def forward(self, x, rev=False):

        if not rev:
            out = self.inv1(x)
            out = self.inv2(out)
            out = self.inv3(out)
            out = self.inv4(out)
            out = self.inv5(out)
            out = self.inv6(out)
            out = self.inv7(out)
            out = self.inv8(out)

            out = self.inv9(out)
            out = self.inv10(out)
            out = self.inv11(out)
            out = self.inv12(out)
            out = self.inv13(out)
            out = self.inv14(out)
            out = self.inv15(out)
            out = self.inv16(out)

        else:
            out = self.inv16(x, rev=True)
            out = self.inv15(out, rev=True)
            out = self.inv14(out, rev=True)
            out = self.inv13(out, rev=True)
            out = self.inv12(out, rev=True)
            out = self.inv11(out, rev=True)
            out = self.inv10(out, rev=True)
            out = self.inv9(out, rev=True)

            out = self.inv8(out, rev=True)
            out = self.inv7(out, rev=True)
            out = self.inv6(out, rev=True)
            out = self.inv5(out, rev=True)
            out = self.inv4(out, rev=True)
            out = self.inv3(out, rev=True)
            out = self.inv2(out, rev=True)
            out = self.inv1(out, rev=True)

        return out
