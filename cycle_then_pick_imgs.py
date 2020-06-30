import cv2
import os
import shutil
directory = "generated_img_2"
if not(os.path.exists(directory)):
    print("given directory is empty/ doesnt exist")
    exit()

save_dir = "checked_generated_dir_2"
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.mkdir(save_dir)

#gettinf all the images to check
image_count = 0
IMAGES = []
for filename in os.listdir(directory):
    if filename.endswith(".jpg") : 
        #print(filename)
        IMAGES.append([os.path.join(directory,  filename), filename])
        image_count +=1
        continue
    else:
        continue

checked = 0
for image_path in IMAGES:
    #print(image_path)
    img = cv2.imread(image_path[0])
    
    cv2.imshow("pick",img)
    key = cv2.waitKey(0) & 0xFF
    

    if int(key) == ord('q') or int(key) == ord('Q'):
        cv2.destroyAllWindows()
        break

    if int(key) == ord('a') or int(key) == ord('A'):
        print("let it pass")

    elif int(key) == ord('d') or int(key) == ord('D'):
        print("save")
        cv2.imwrite(os.path.join(save_dir, image_path[1]), img)

    checked += 1
    print(str(checked) + "/" + str(image_count ))
    
