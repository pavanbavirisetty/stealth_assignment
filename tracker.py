import cv2
import time
import numpy as np
    
def track_basketball(video_path):
            # read the video 
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(3))
            height = int(cap.get(4))
            cap.set(cv2.CAP_PROP_FPS, 30)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('output.avi', fourcc, 20.0, (width, height))
            
            # initialization 
            dribble_count = 0
            left_hand_dribble_count = 0
            right_hand_dribble_count = 0
            last_dribble_time = time.time()
            dribble_time_window = 0.35
            
            line_x = int(width / 2)
            line_y = int((height/2)+120)
            line_color = (255, 0, 0)
            line_thickness = 2
            

            while True:
                ret, frame = cap.read()

                if not ret:
                    break
                
                # identifying the ball in the frame 
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                lower_yellow = np.array([20, 100, 100])
                upper_yellow = np.array([30, 255, 255])

                mask = cv2.inRange(hsv, lower_yellow, upper_yellow)


                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

                for contour in contours:
                    area = cv2.contourArea(contour) 
                    if area > 500:
                        x, y, w, h = cv2.boundingRect(contour)
                        center = (int(x + w / 2), int(y + h / 2))
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.circle(frame, center, 5, (0, 0, 255), -1)
                        
                        # splitting the frame into parts to count the dribbles effectively 
                        cv2.line(frame, (0, line_y), (width, line_y), line_color, line_thickness)
                        cv2.line(frame, (line_x, 0), (line_x, 2*height), (255, 0, 0), line_thickness)
                        
                        # counting the total no of dribbles 
                        if center[1] >= line_y and time.time() - last_dribble_time >= dribble_time_window:
                            cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), line_thickness)
                            last_dribble_time = time.time()
                            dribble_count += 1
                            # print(dribble_count)
                            
                            # counting the no of dribbles with each hand
                            if center[0] < width / 2:
                                cv2.line(frame, (line_x, 0), (line_x, 2*height), (0, 0, 0), line_thickness)
                                right_hand_dribble_count += 1
                            else:
                                cv2.line(frame, (line_x, 0), (line_x, 2*height), (255, 255, 255), line_thickness)
                                left_hand_dribble_count += 1

                out.write(frame)

                cv2.imshow('Basketball Tracking', frame)

                key = cv2.waitKey(30)

                if key == 27:
                    break

            cap.release()
            cv2.destroyAllWindows()
            out.release()
            result = print(f"Total No of Dribbles: {dribble_count}\nLeft Hand Dribbles: {left_hand_dribble_count}\nRight Hand Dribbles: {right_hand_dribble_count}\n")
            
            return result
