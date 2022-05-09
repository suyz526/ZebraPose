import torch.nn as nn
import torch
from model.aspp import ASPP, ASPP_50, ASPP_non_binary_ablationstudy
from model.resnet import ResNet50_OS8, ResNet34_OS8

#################### loss ######################
################################################
class BinaryCodeLoss(nn.Module):
    def __init__(self, binary_code_loss_type, mask_binary_code_loss, divided_number_each_iteration, use_histgramm_weighted_binary_loss=False):
        super().__init__()
        self.binary_code_loss_type = binary_code_loss_type
        self.mask_binary_code_loss = mask_binary_code_loss
        self.divided_number_each_iteration = divided_number_each_iteration
        self.use_histgramm_weighted_binary_loss = use_histgramm_weighted_binary_loss

        if self.use_histgramm_weighted_binary_loss: # this Hammloss will be used in both case, for loss, or for histogramm
            self.hamming_loss = HammingLoss()

        if binary_code_loss_type == "L1":
            self.loss = nn.L1Loss(reduction="mean")
        elif binary_code_loss_type == "BCE":
            self.loss = nn.BCEWithLogitsLoss(reduction="mean")
        elif binary_code_loss_type == "CE":
            self.loss = nn.CrossEntropyLoss(reduction="mean")
        else:
            raise NotImplementedError(f"unknown mask loss type: {binary_code_loss_type}")

        if self.use_histgramm_weighted_binary_loss:
            assert binary_code_loss_type == "BCE"  # currently only have the implementation with BCEWithLogitsLoss
            self.loss= BinaryLossWeighted(nn.BCEWithLogitsLoss(reduction='none'))

        self.histogram= None
        
    def forward(self, pred_binary_code, pred_mask, groundtruth_code):
        ## calculating hamming loss and bit error histogram for loss weighting
        if self.use_histgramm_weighted_binary_loss:
            loss_hamm, histogram_new = self.hamming_loss(pred_binary_code, groundtruth_code, pred_mask.clone().detach())
            if self.histogram is None:
                self.histogram  = histogram_new
            else:
                self.histogram = histogram_new*0.05+self.histogram*0.95
            
            ## soft bin weigt decrease 
            hist_soft = torch.minimum(self.histogram,0.51-self.histogram)
            bin_weights = torch.exp(hist_soft*3).clone()    

        if self.mask_binary_code_loss:
            pred_binary_code = pred_mask.clone().detach() * pred_binary_code

        if self.binary_code_loss_type == "L1":
            pred_binary_code = pred_binary_code.reshape(-1, 1, pred_binary_code.shape[2], pred_binary_code.shape[3])
            pred_binary_code = torch.sigmoid(pred_binary_code)
            groundtruth_code = groundtruth_code.view(-1, 1, groundtruth_code.shape[2], groundtruth_code.shape[3])
        elif self.binary_code_loss_type == "BCE" and not self.use_histgramm_weighted_binary_loss:
            pred_binary_code = pred_binary_code.reshape(-1, pred_binary_code.shape[2], pred_binary_code.shape[3])
            groundtruth_code = groundtruth_code.view(-1, groundtruth_code.shape[2], groundtruth_code.shape[3])
        elif self.binary_code_loss_type == "CE":
            pred_binary_code = pred_binary_code.reshape(-1, self.divided_number_each_iteration, pred_binary_code.shape[2], pred_binary_code.shape[3])
            groundtruth_code = groundtruth_code.view(-1, groundtruth_code.shape[2], groundtruth_code.shape[3])
            groundtruth_code = groundtruth_code.long()
        
        if self.use_histgramm_weighted_binary_loss:
            loss = self.loss(pred_binary_code, groundtruth_code, bin_weights)
        else:
            loss = self.loss(pred_binary_code, groundtruth_code)
    
        return loss


class BinaryLossWeighted(nn.Module):
    def __init__(self, baseloss):
        #the base loss should have reduction 'none' 
        super().__init__()
        self.base_loss = baseloss

    def forward(self,input,target,weight):
        base_output=self.base_loss(input,target)
        assert base_output.ndim == 4
        output = base_output.mean([0,2,3])
        output = torch.sum(output*weight)/torch.sum(weight)
        return output


class MaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss(reduction="mean")
       
    def forward(self, pred_mask, groundtruth_mask): 
        pred_mask = pred_mask[:, 0, :, :]
        pred_mask = torch.sigmoid(pred_mask)
        
        return self.loss(pred_mask, groundtruth_mask)


class HammingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,predicted_code_prob,GT_code,mask):
        assert predicted_code_prob.ndim ==4
        mask_hard  = mask.round().clamp(0,1) # still kept round and clamp for safety
        code1_hard = torch.sigmoid(predicted_code_prob).round().clamp(0,1)
        code2_hard = GT_code.round().clamp(0,1) # still kept round and clamp for safety
        hamm = torch.abs(code1_hard-code2_hard)*mask_hard
        histogram = hamm.sum([0,2,3])/(mask_hard.sum()+1)
        hamm_loss = histogram.mean()
        
        return hamm_loss,histogram.detach()


#################### model ######################
################################################

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
}


class BinaryCodeNet_Deeplab(nn.Module):
    def __init__(
        self, 
        num_resnet_layers, 
        binary_code_length, 
        divided_number_each_iteration,
        concat=False, 
        output_kernel_size = 1, 
    ):
        super(BinaryCodeNet_Deeplab, self).__init__()
        self.concat = concat

        # DeepLabV3 and DeepLabV3_non_binary can be merged as 1 model. To ensure the checkpoint works, we keep it unchanged for now.
        if divided_number_each_iteration == 2:
            # hard coded 1 for object mask
            # we assumed for binary case, no CE loss will be used. Otherwise it will be binary_code_length*2 + 1
            self.net = DeepLabV3(num_resnet_layers, binary_code_length + 1, concat=self.concat, output_kernel_size=output_kernel_size)
        else:
            self.net = DeepLabV3_non_binary(num_resnet_layers, binary_code_length=binary_code_length, divided_number_each_iteration=divided_number_each_iteration, concat=self.concat, output_kernel_size=output_kernel_size)

    def forward(self, inputs):
        return self.net(inputs)


class DeepLabV3(nn.Module):
    def __init__(self, num_resnet_layers, num_classes, concat=False, output_kernel_size=1):
        super(DeepLabV3, self).__init__()

        self.num_classes = num_classes
        self.concat = concat
        self.num_resnet_layers = num_resnet_layers
        if num_resnet_layers == 34:
            self.resnet = ResNet34_OS8(34, concat) # NOTE! specify the type of ResNet here
            self.aspp = ASPP(num_classes=self.num_classes, concat=concat, output_kernel_size=output_kernel_size) 
        elif num_resnet_layers == 50:
            self.resnet = ResNet50_OS8(50, concat) # NOTE! specify the type of ResNet here
            self.aspp = ASPP_50(num_classes=self.num_classes, concat=concat, output_kernel_size=output_kernel_size) 
        

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))
        if not self.concat:
            feature_map = self.resnet(x) # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8). If self.resnet is ResNet50-152, it will be (batch_size, 4*512, h/16, w/16))
            output = self.aspp(feature_map) # (shape: (batch_size, num_classes, h/16, w/16))
        else:
            x_high_feature, x_128, x_64, x_32, x_16 = self.resnet(x)
            output = self.aspp(x_high_feature, x_128, x_64, x_32, x_16)

        #output = F.interpolate(output, size=(h, w), mode="bilinear") # (shape: (batch_size, num_classes, h, w))

        mask,binary_code = torch.split(output,[1,self.num_classes-1],1)
        return mask, binary_code


class DeepLabV3_non_binary(nn.Module):
    def __init__(self, num_resnet_layers, binary_code_length=16, divided_number_each_iteration=2, concat=False, output_kernel_size=1):
        super(DeepLabV3_non_binary, self).__init__()

        self.concat = concat
        self.resnet = ResNet34_OS8(34, concat) # NOTE! specify the type of ResNet here
        self.aspp = ASPP_non_binary_ablationstudy(code_output_length=binary_code_length, divided_number_each_iteration=divided_number_each_iteration, concat=concat, output_kernel_size=output_kernel_size) # NOTE! if you use ResNet50-152, set self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) instead

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))
        if not self.concat:
            feature_map = self.resnet(x) # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8). If self.resnet is ResNet50-152, it will be (batch_size, 4*512, h/16, w/16))
            mask, binary_code = self.aspp(feature_map) # (shape: (batch_size, num_classes, h/16, w/16))
        else:
            x_high_feature, x_128, x_64, x_32, x_16 = self.resnet(x)
            mask, binary_code = self.aspp(x_high_feature, x_128, x_64, x_32, x_16)

        #output = F.interpolate(output, size=(h, w), mode="bilinear") # (shape: (batch_size, num_classes, h, w))

        return mask, binary_code
    
    

