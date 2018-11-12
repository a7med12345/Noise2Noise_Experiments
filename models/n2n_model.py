import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from math import log10

class N2nModel(BaseModel):
    def name(self):
        return 'N2nModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper
        # (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')
        parser.set_defaults(dataset_mode='t2t')
        parser.set_defaults(netG='unet_256')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G1','G2','PSNR_clean','PSNR_n2n','PSNR_original']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A1','real_A2', 'fake_B_n','fake_B_c','real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks

        self.model_names = ['G1','G2']

        # load/define networks
        self.netG1 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG2 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterion = torch.nn.MSELoss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G1 = torch.optim.Adam(self.netG1.parameters(),
                                         lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G2 = torch.optim.Adam(self.netG2.parameters(),
                                                 lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G1)
            self.optimizers.append(self.optimizer_G2)



    def set_input(self, input):

        self.real_A1 = input['A1'].to(self.device)
        self.real_A2 = input['A2'].to(self.device)
        self.real_B = input['B' ].to(self.device)
        self.image_paths = input['A1_paths']


    def forward(self):
        self.fake_B_n = self.netG1(self.real_A1)
        self.fake_B_c = self.netG2(self.real_A1)


    def backward(self):

        self.loss_G1 = self.criterion( self.fake_B_n,self.real_A2)
        self.loss_G2 = self.criterion( self.fake_B_c, self.real_B)


        self.loss_PSNR_clean = 10 * log10(1 / self.loss_G2.detach().item())
        self.loss_PSNR_n2n = 10 * log10(1 / self.criterion( self.fake_B_n.detach(),self.real_B.detach()).item())
        self.loss_PSNR_original = 10 * log10(1 / self.criterion(self.real_A1.detach(), self.real_B.detach()).item())

        self.loss_G1.backward()
        self.loss_G2.backward()



    def optimize_parameters(self):
        self.forward()
        self.optimizer_G1.zero_grad()
        self.optimizer_G2.zero_grad()
        self.backward()
        self.optimizer_G1.step()
        self.optimizer_G2.step()

