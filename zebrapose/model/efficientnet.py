import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.efficientnet import MBConvConfig,_efficientnet_conf,MBConv
import copy
from functools import partial
from typing import List
from torchvision.ops.misc import ConvNormActivation



class efficientnet_intermediate_out(nn.Module):
	def __init__(self,version,concat):
		super(efficientnet_intermediate_out,self).__init__()

		self.concat_decoder = concat
		
		if version == 'b5':
			print("Pretrained efficientnet_b5")
			
			efficientnet = models.efficientnet_b5()
			dirname = os.path.dirname(__file__)
			filename = os.path.join(dirname, '../pretrained_backbone/efficientnet/efficientnet_b5_lukemelas-b6417697.pth')
			efficientnet.load_state_dict(torch.load(filename))

			#remove fully connected , avg pool
			self.efficientnet = nn.Sequential(*list(efficientnet.children())[0])

			

			if self.concat_decoder == True:
			 					
				self.eff_layer_2 = nn.Sequential(*list(self.efficientnet.children())[:2])
				
				self.eff_layer_3 = nn.Sequential(*list(self.efficientnet.children())[2:3])
				
				self.eff_layer_4 = nn.Sequential(*list(self.efficientnet.children())[3:4])
				
				self.eff_layer_6 = nn.Sequential(*list(self.efficientnet.children())[4:6])
				
				self.eff_layer_9 = nn.Sequential(*list(self.efficientnet.children())[6:9])


		else:
			raise Exception("Not implemented or invalid!!")


	def forward(self,x):
		if self.concat_decoder:
			
			l2 = self.eff_layer_2(x)
			
			l3 = self.eff_layer_3(l2)
			
			l4 = self.eff_layer_4(l3)
			
			l6 = self.eff_layer_6(l4)
			
			l9 = self.eff_layer_9(l6)

		return l9,l6,l4,l3,l2


class efficientnet_upsampled(nn.Module):
	def __init__(self,version,concat):
		super(efficientnet_upsampled,self).__init__()

		self.concat_decoder = concat
		
		if True:
			print("Pretrained efficientnet_b4")
			efficientnet = models.efficientnet_b4()
			dirname = os.path.dirname(__file__)
			filename = os.path.join(os.path.dirname(dirname), 'pretrained_backbone/efficientnet/efficientnet_b4_rwightman-7eb33cd5.pth')
			efficientnet.load_state_dict(torch.load(filename))

			#remove fully connected , avg pool
			self.efficientnet = nn.Sequential(*list(efficientnet.children())[0])

			block = MBConv
			norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
			layers: List[nn.Module] = []
			width_mult = 1.4
			depth_mult=1.8

			stochastic_depth_prob = 0.2

			bneck_conf = partial(MBConvConfig, width_mult=width_mult, depth_mult=depth_mult)
			inverted_residual_setting = [
								bneck_conf(6, 3, 1, 40, 80, 3),
								bneck_conf(6, 5, 1, 80, 112, 3),
								bneck_conf(6, 5, 1, 112, 192, 4),
								bneck_conf(6, 3, 1, 192, 320, 1),
								]

			if self.concat_decoder == True:

				
				self.eff_layer_2 = nn.Sequential(*list(self.efficientnet.children())[:2])
				self.eff_layer_3 = nn.Sequential(*list(self.efficientnet.children())[2:3])
				self.eff_layer_4 = nn.Sequential(*list(self.efficientnet.children())[3:4])
				
				total_stage_blocks = sum([cnf.num_layers for cnf in inverted_residual_setting])
				stage_block_id = 0
				for cnf in inverted_residual_setting:
					stage: List[nn.Module] = []
					for _ in range(cnf.num_layers):
						# copy to avoid modifications. shallow copy is enough
						block_cnf = copy.copy(cnf)

						# overwrite info if not the first conv in the stage
						if stage:
							block_cnf.input_channels = block_cnf.out_channels
							block_cnf.stride = 1

						# adjust stochastic depth probability based on the depth of the stage block
						sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

						stage.append(block(block_cnf, sd_prob, norm_layer))
						stage_block_id += 1

					layers.append(nn.Sequential(*stage))
				lastconv_input_channels = inverted_residual_setting[-1].out_channels
				lastconv_output_channels =  lastconv_input_channels
				layers.append(ConvNormActivation(lastconv_input_channels, lastconv_output_channels, kernel_size=1,
										 norm_layer=norm_layer, activation_layer=nn.SiLU))

				self.final_layer = nn.Sequential(*layers)

		else:
			raise Exception("Not implemented or invalid!!")


	def forward(self,x):
		if self.concat_decoder:
			#l1 = self.eff_layer_1(x)
			
			l2 = self.eff_layer_2(x)
			
			l3 = self.eff_layer_3(l2)
			
			l4 = self.eff_layer_4(l3)
			
			final = self.final_layer(l4)
			
		return final,l3,l2