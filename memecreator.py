import cv2
import nodes
import torch
from PIL import Image
import torchvision.transforms as Transforms
import torch.nn.functional as F   
import numpy as np
import random

#  TK Collector - Custom Node for Meme Creation
#  August 9, 2025
#  https://github.com/trashkollector/TKMemeCreator
#  https://civitai.com/user/trashkollector175
    
class TKMemeCreator:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "image": ("IMAGE",),
                "audio": ("AUDIO",),
                "frame_count":  ("INT",{"default": 85, "min": 40}),
                "fps":  ("FLOAT",{"default": 16, "min": 16}),
                "top_text": ("STRING",{"default": "Top Text"}),
                "bottom_text": ("STRING",{"default": "Bottom Text"}),
                "num_secs_delay_end_text":  ("INT",{"default": 2, "min": 0}),
                "wobble_text": ("BOOLEAN",),
                "color_start_text": (["red", "white","black","blue","yellow", "green"],),
                "color_end_text": (["red", "white","black","blue","yellow", "green"],),
            },
        }

    RETURN_TYPES = ("IMAGE","AUDIO")
    RETURN_NAMES = ("image","audio")

    FUNCTION = "tkmemecreator"

    #OUTPUT_NODE = False

    CATEGORY = "TKMemeCreator"

    # Assume 'video_tensor' is your video data as a PyTorch tensor
    # Format: [T, H, W, C] (Time, Height, Width, Channels)
    # Data type should be uint8, with values in the range [0, 255]
    #  number of channels (e.g., 3 for RGB)
    # To represent a video, NumPy extends this concept by adding an extra dimension for the frames.
    #    This results in a 4D array with the shape (frames, height, width, channels).
    
    def tkmemecreator(self, image, audio, frame_count, fps, top_text, bottom_text, num_secs_delay_end_text, wobble_text, color_start_text, color_end_text):
        tensor_shape = image.shape
        height = tensor_shape[1]
        width = tensor_shape[2]
        
        colorTop = (0, 255, 0)
        colorBottom = (0, 255, 0)
         
        font = cv2.FONT_HERSHEY_TRIPLEX 
        scale = 2
        thickness=5
        
        if (width < 500) :
           scale =0.7
           thickness=1
        elif (width < 700) :
           scale = 1.2
           thickness=3
        elif (width < 900) :
           scale = 1.6
           thickness=4
           
        #center
        (text_width, text_height), baseline = cv2.getTextSize(top_text, font, scale, thickness)
        x=0
        if (text_width < width) :
           x = (width - text_width) /2 
        text_position_top = ( int(x)  ,50)
        
        (text_width, text_height), baseline = cv2.getTextSize(bottom_text, font, scale, thickness)
        x=0
        if (text_width < width) :
           x = (width - text_width) /2 
        text_position_bottom = ( int(x)  ,height - 60)
        
        
        # Generate a random waveform function along the y-axis - used for wobble
        y_trans = self.generate_translation(frame_count, 16, tuple([0.5, 2.5]), tuple([5.0,10.0]))
        x_trans = self.generate_translation(frame_count, 16, tuple([0.0, 0.8]), tuple([1.0,5.0]))
              
        video_numpy = (image.numpy() * 255).astype(np.uint8)  #convert video
   
    


     
        print("Applying Meme")
     
        i=0
        frames=[]
        for frame in video_numpy: 
            if (i >= frame_count-1) :
                break
                
            phase2=False
            if    int(  float(i) / fps ) >= num_secs_delay_end_text :
                phase2=True

            if (wobble_text) :
                if phase2 :
                    frame = self.printText(frame, 
                                           top_text, 
                                           ( text_position_top[0], text_position_top[1] ), 
                                           font, 
                                           scale, 
                                           self.getColor(color_start_text), 
                                           thickness, 
                                           color_start_text)
                else :
                    frame = self.printText(frame, 
                                           top_text, 
                                           (text_position_top[0]+ int(x_trans[i]), text_position_top[1]+ int(y_trans[i])), 
                                           font, 
                                           scale, 
                                           self.getColor(color_start_text), 
                                           thickness, 
                                           color_start_text)
            
                if   phase2 :
                     frame = self.printText(frame, 
                                            bottom_text, 
                                           (text_position_bottom[0]+ int(x_trans[i]), text_position_bottom[1]+ int(y_trans[i])), 
                                           font, 
                                           scale,
                                           self.getColor(color_end_text), 
                                           thickness, 
                                           color_end_text)
            else :
                frame = self.printText(frame, top_text, text_position_top, font, scale, 
                                       self.getColor(color_start_text), thickness, color_start_text)
            
                if    int(  float(i) / fps ) >= num_secs_delay_end_text :
                     frame = self.printText(frame, bottom_text, text_position_bottom, 
                                font, scale, self.getColor(color_end_text), thickness, color_end_text)    

            frames.append(frame)
            
          
                    
            i = i+1
            
            
            
        # Convert to tensor
        numpy_array = np.array(frames)
        theTensor = torch.from_numpy(numpy_array)
        theTensor = theTensor.float()  / 255.0
        
        print(theTensor.shape)
            
        return (theTensor,audio)
        
    def printText(self,frame, text, text_position, font, scale, color, thickness, colorStr) :
        
         outcol =(0,0,0)
         if (colorStr == "black" or colorStr=="blue") :
             outcol=(255,255,255)
             
         frame = cv2.putText(frame, text, text_position, font, scale, outcol, thickness+ thickness)
         
         frame = cv2.putText(frame, text, text_position, font, scale, color, thickness)
         
         return frame
    
    def getColor(self, colorStr) :
        if colorStr == "white" :
            return (255,255,255)
        elif colorStr =="black" :
            return (0,0,0)
        elif colorStr =="red" :
            return (255,0,0)
        elif colorStr =="blue" :
            return (0,0,255)
        elif colorStr =="green" :
            return (0,255,0)
        elif colorStr =="yellow" :
            return (255,255,0)    
      
    def generate_translation(
        self,
        n_frames: int, 
        fps: float, 
        amplitudes: tuple[float, float],
        frequencies: tuple[float, float]
    ) -> list:
        """
        Generate a list of translation values to create a hand shaking effect along one 
        axis in an 80-second video.

        Parameters:
            n_frames (int): The total number of frames in the video.
            fps (float): Frames per second of the video.
            amplitudes (tuple[float, float]): The amplitudes of the sinusoidal waves 
                                              along y-axis.
            frequencies (tuple[float, float]): The frequencies of the sinusoidal waves 
                                               along the x-axis.

        Returns:
            list: A list of translation values for each frame to create the hand shaking
                  effect.
        """
        

        num_points = 2000
        num_waves = np.random.randint(30, 40)
        amplitude_min, amplitude_max = amplitudes
        frequence_min, frequence_max = frequencies
        
        x = np.linspace(0, 10, num_points)
        y = np.zeros_like(x)
        
        for _ in range(num_waves):
            frequency = np.random.uniform(frequence_min, frequence_max)
            amplitude = np.random.uniform(amplitude_min, amplitude_max)
            phase_shift = np.random.uniform(0, 2*np.pi)
            y += amplitude * np.sin(2*np.pi*frequency*x + phase_shift)

        duration = n_frames / fps
        fixed = int(num_points * duration / 80)

        x_fixed, y_fixed = x[:fixed], y[:fixed]

        x_interpolated = np.linspace(0, max(x_fixed), n_frames)
        y_interpolated = np.interp(x_interpolated, x_fixed, y_fixed)

              
        return y_interpolated    

    
     
        
        
    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """
    
    
    
    
    
    #@classmethod
    #def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
    #    return ""

# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"


# Add custom API routes, using router
from aiohttp import web
from server import PromptServer

@PromptServer.instance.routes.get("/tkmemecreator")
async def get_tkmemecreator(request):
    return web.json_response("tkmemecreator")


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "TKMemeCreator": TKMemeCreator
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "TKMemeCreator": "TKMemeCreator"
}
