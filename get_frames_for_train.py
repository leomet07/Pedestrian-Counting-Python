import cv2
import os
import shutil
#cap = cv2.VideoCapture("rtsp://user2:user#55@46.52.182.186:37778/cam/realmonitor?channel=2&subtype=0")
cap = cv2.VideoCapture("lomg.mp4")
write = True

print("writing")


#fps = 15
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#output_movie = cv2.VideoWriter('output_for_train.mp4', fourcc, fps, (w, h))
current_frame_count = 0
saved_img = 0
x= 820
y = 66
w = 468
h = 800

img_dir = "generated_img_2"

if os.path.exists(img_dir):
    shutil.rmtree(img_dir)

os.mkdir(img_dir)
totalfps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        #print("Could not be read")
        break

    current_frame_count += 1

    

    #frame = cv2.resize(frame, (720 , 480))
    input_frame = (frame.copy())[y:y + h, x:x + w]
    image_height, image_width, _ = input_frame.shape
    #print(current_frame)
    #print(input_frame.shape)
    if current_frame_count % 32 == 0:
        #every 50 frames, save
        print("saving " + str(current_frame_count) + "/" + str(totalfps))
        saved_img += 1
        cv2.imwrite(os.path.join(img_dir,str(saved_img) + ".jpg"), input_frame)
        
    #output_movie.write(input_frame)
    '''
    cv2.imshow("window",input_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    '''