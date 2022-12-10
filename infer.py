import numpy as np
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import cv2

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1
# Patch the location of gfile
tf.gfile = tf.io.gfile
# List of the strings that is used to add correct label for each box.
category_index = label_map_util.create_category_index_from_labelmap(r'.\labels\label_map.pbtxt', use_display_name=True)

#CHANGE WHICH MODEL TO USE HERE
#detection_model = tf.saved_model.load(str(r".\models\person\saved_model"))
detection_model = tf.saved_model.load(str(r".\models\BACKUP\saved_model"))
#CHANGE WHICH VIDEO TO USE HERE
videoTitle = "cursed.mp4"

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors, which are converted to numpy arrays and the batch dimension is removed.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict
  
def show_inference(model, image_np):
    output_dict = run_inference_for_single_image(model, image_np)
    # Visualization of the results of a detection.
    final_img =( 
        vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8))
    return final_img
    
videoPath = ".\\testvideos\\"
videoFinal = videoPath + videoTitle
resultPath = ".\\resultvideos\\"
resultString = resultPath + videoTitle + '.avi'

cap = cv2.VideoCapture((videoFinal))
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))   
size = (frame_width, frame_height)
result = cv2.VideoWriter(resultString, 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
                         
while 1:
    ret,img = cap.read()
    if ret == False:
        break
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    final_img = show_inference(detection_model, img)
    final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
    result.write(final_img)
    # uncomment if you want to show live boundary boxes 
    cv2.imshow('img', final_img)    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

"""
Test with forcing a 640x640 resize into the inference
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 5, (640,640))
while 1:
    _,img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    b = cv2.resize(img,(640,640),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
    final_img = show_inference(detection_model, b)
    final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)    
    cv2.imshow('img', final_img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
"""        
