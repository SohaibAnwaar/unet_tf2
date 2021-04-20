
from PIL import Image, ImageDraw
import json, os

def convert_json_to_mask(json_path, base_path, save_masks = ''):
    '''
    Description:
        This Generator will help you out in converting your annotations from polygon to mask image
        so that you can use it for the training. I done annotaions with VIA Image annotator with
        polygons and make this fucntion to convert those annotations to the mask so that I can use
        it to train my unet model
        
    Input:
        json_path  (str) : Path of the json in which you saved your annnotations
        base_path  (str) : Base path where your images are located
        save_masks (str) : Location where you want to save the masks
        
    Output:
        Image      (nd_array) : Numpy array of original Image
        Mask       (nd_array) : Numpy array of mask image
    
    
    '''
    
    with open(json_path) as f:
        data = json.load(f)

    for i in data:
        
        # Getting data from the json
        image_name = data[i]['filename']
        save_mask_here = f"{save_masks}{image_name}"
        if not os.path.isfile(save_mask_here):
        
            temp       = data[i]['regions']['0']['shape_attributes']
            x_points   = temp['all_points_x']
            y_points   = temp['all_points_y']
            xy_points  = [i for i in zip(x_points,y_points)] 

            # Making masks from the json file
            original_img = Image.open(f"{base_path}{image_name}")
            image_size = original_img.size 
            img = Image.new("RGB", image_size, "black") 
            img1 = ImageDraw.Draw(img)  
            img1.polygon(xy_points, fill ="#f9f9f9") 

            if save_masks != '':
                img.save(save_mask_here)
            yield original_img, img
    
        else: yield None, None
            
        