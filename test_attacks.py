import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_single import Face_detect_crop
from util.reverse2original import reverse2wholeimage
import os
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet

import latent_code_attack

def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0

transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

inverse_transformer = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])




def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

if __name__ == '__main__':
    opt = TestOptions().parse()

    start_epoch, epoch_iter = 1, 0
    crop_size = opt.crop_size

    torch.nn.Module.dump_patches = True
    if crop_size == 512:
        opt.which_epoch = 550000
        opt.name = '512'
        mode = 'ffhq'
    else:
        mode = 'None'
    logoclass = watermark_image('./simswaplogo/simswaplogo.png')
    model = create_model(opt)
    model.eval()

    spNorm =SpecificNorm() #!!!作用是什么？
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    # app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640),mode=mode)
    #降低threshold，防止detect返回None
    app.prepare(ctx_id= 0, det_thresh=0.1, det_size=(640,640),mode=mode)


    # with torch.no_grad():
    # pic_a = opt.pic_a_path #source (identity) face

    source_dir = opt.source_dir #source faces dir (攻击目标)
    target_dir = opt.target_dir #target faces dir 

    source_list = []
    target_list = []

    for root,dirs,files in os.walk(source_dir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                source_list.append(os.path.join(root,file))

    for root,dirs,files in os.walk(target_dir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                target_list.append(os.path.join(root,file))

    #排序保证每次运行顺序一致
    source_list.sort()
    target_list.sort()
    

    idx = 1
    for i in range(len(source_list)):
        pic_a = source_list[i]

        img_a_whole = cv2.imread(pic_a)
        # print(img_a_whole.shape)

        # img_a_align_crop, _ = app.get(img_a_whole,crop_size)#face detect and crop
        
        #！！！new edit: 不需要detect，直接resize到crop_size
        img_a_align_crop = cv2.resize(img_a_whole, (crop_size,crop_size))

        # print("img_a_align_crop shape:",img_a_align_crop.shape)
        
        # img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0],cv2.COLOR_BGR2RGB))
        img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop,cv2.COLOR_BGR2RGB)) 
        # img_a = transformer_Arcface(img_a_align_crop_pil) #[3,224,224]
        
        #不进行transform normalize, 直接转换为tensor
        img_a = transforms.ToTensor()(img_a_align_crop_pil)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2]) #[1,3,224,224]

        img_id = img_id.cuda()

        #create latent id
        img_id_downsample = F.interpolate(img_id, size=(112,112))
        latend_id = model.netArc(img_id_downsample) #use Arcface to generate latent id
        latend_id = F.normalize(latend_id, p=2, dim=1) #[1,512]

        """
        latent code attack
        """
        # Initialize Metrics
        l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
        n_dist, n_samples = 0, 0

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        y = latend_id.cpu().detach()

        attack = latent_code_attack.IFGSMAttack(model, device=device)

        #传入img_id作为原始X, y作为目标Y，返回攻击后的adv_img_id
        adv_img_id,perturb = attack.perturb(img_id.clone().detach_(), y)

        #根据adv_img_id提取出adv_latend_id
        adv_img_id_downsample = F.interpolate(adv_img_id, size=(112,112))
        adv_latend_id = model.netArc(adv_img_id_downsample)
        adv_latend_id = F.normalize(adv_latend_id, p=2, dim=1) #[1,512]



        #******保存img_id******
        img_id_np = img_id.cpu().detach()
        img_id_np = img_id_np[0] #[3,224,224]
        # img_id_np = inverse_transformer(img_id_np)
        img_id_np = img_id_np.numpy()
        img_id_np = np.transpose(img_id_np,(1,2,0)) #[224,224,3]
        # print("min(img_id_np):",img_id_np.min())
        # print("max(img_id_np):",img_id_np.max())
        img_id_np = img_id_np*255
        img_id_np = img_id_np.astype(np.uint8)
        img_id_np = cv2.cvtColor(img_id_np,cv2.COLOR_RGB2BGR)
        cv2.imwrite('/content/output/ori_source/{}.jpg'.format(idx),img_id_np)



        #******保存adv_img_id******
        adv_img_id_pil = adv_img_id.cpu().detach() #[1,3,224,224]
        adv_img_id_pil = adv_img_id_pil[0] # [3,224,224]
        # adv_img_id_pil = inverse_transformer(adv_img_id_pil)
        adv_img_id_pil = adv_img_id_pil.numpy()
        adv_img_id_pil = np.transpose(adv_img_id_pil,(1,2,0)) #[224,224,3]
        # print("min(adv_img_id_pil):",adv_img_id_pil.min())
        # print("max(adv_img_id_pil):",adv_img_id_pil.max())
        adv_img_id_pil = adv_img_id_pil*255
        adv_img_id_pil = adv_img_id_pil.astype(np.uint8)
        adv_img_id_pil = cv2.cvtColor(adv_img_id_pil,cv2.COLOR_RGB2BGR)
        cv2.imwrite('/content/output/adv_source/{}.jpg'.format(idx),adv_img_id_pil)


        ############## Forward Pass ######################
        for j in range(len(target_list)):
            pic_b = target_list[j]
            img_b_whole = cv2.imread(pic_b)
            # img_b_whole = cv2.resize(img_b_whole, (800, 800))

            # there might be multiple faces in the target face
            img_b_align_crop_list, b_mat_list = app.get(img_b_whole,crop_size)#face detect and crop
            # detect_results = None
            swap_result_list = []

            b_align_crop_tenor_list = []

            for b_align_crop in img_b_align_crop_list:
                #b_align_crop shape: [224,224,3]
                b_align_crop_tenor = _totensor(cv2.cvtColor(b_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()

                #!!!!input id_vector, target face; output result image
                swap_result = model(None, b_align_crop_tenor, latend_id, None, True)[0] #[3,224,224]
                
                swap_result_save = swap_result.cpu().detach().numpy()
                # print("min:",np.min(swap_result_save),"max:",np.max(swap_result_save))
                swap_result_save = swap_result_save.transpose(1,2,0)
                swap_result_save = swap_result_save * 255
                swap_result_save = swap_result_save.astype(np.uint8)
                swap_result_save = cv2.cvtColor(swap_result_save,cv2.COLOR_RGB2BGR)
                # print("swap_result shape:",swap_result.shape)
                # ！！！临时保存swap_result，用于测试 ！！！
                cv2.imwrite('/content/output/swap_result/{}.jpg'.format(idx),swap_result_save)

                swap_result_list.append(swap_result)
                b_align_crop_tenor_list.append(b_align_crop_tenor)

                #生成adv_swap_result
                adv_swap_result = model(None, b_align_crop_tenor, adv_latend_id, None, True)[0] #[3,224,224]
                adv_swap_result_save = adv_swap_result.cpu().detach().numpy()
                adv_swap_result_save = adv_swap_result_save.transpose(1,2,0)
                adv_swap_result_save = adv_swap_result_save * 255
                adv_swap_result_save = adv_swap_result_save.astype(np.uint8)
                adv_swap_result_save = cv2.cvtColor(adv_swap_result_save,cv2.COLOR_RGB2BGR)
                cv2.imwrite('/content/output/adv_swap_result/{}.jpg'.format(idx),adv_swap_result_save)

            #保存target图片
            img_b_whole_save = img_b_whole.copy()
            img_b_whole_save = cv2.resize(img_b_whole_save, (crop_size, crop_size))
            cv2.imwrite('/content/output/target/{}.jpg'.format(idx),img_b_whole_save)

            

            if opt.use_mask:
                n_classes = 19
                net = BiSeNet(n_classes=n_classes)
                net.cuda()
                save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
                net.load_state_dict(torch.load(save_pth))
                net.eval()
            else:
                net =None
            
            # 将换脸结果替换回 target image中得到result image
            reverse2wholeimage(b_align_crop_tenor_list, swap_result_list, b_mat_list, crop_size, img_b_whole, logoclass, \
                os.path.join(opt.output_path, '{}.jpg'.format(idx)), opt.no_simswaplogo,pasring_model =net,use_mask=opt.use_mask, norm = spNorm)

            # 将adv_swap_result替换回 target image中得到adversarial result image
            reverse2wholeimage(b_align_crop_tenor_list, [adv_swap_result], b_mat_list, crop_size, img_b_whole, logoclass, \
                os.path.join(opt.adv_output_path, '{}.jpg'.format(idx)), opt.no_simswaplogo,pasring_model =net,use_mask=opt.use_mask, norm = spNorm)


            print(' ')
            print("processing {}/{}".format(idx, len(source_list)*len(target_list)))

            idx += 1