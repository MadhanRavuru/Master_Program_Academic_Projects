import numpy as np

def quaternion_angle_metric(q_anchor,q_puller):
  angle = 2 * np.arccos(np.abs(np.dot(q_anchor,q_puller)))
  return angle

def batch_generator(train_data,train_labels,train_class,database_data,database_labels,database_class):
    images_in_triplet = 3  
    temp = np.zeros((train_data.shape[0], images_in_triplet, train_data.shape[1], train_data.shape[2], train_data.shape[3]))
    
    for idx, image in enumerate(train_data):
        
        print(idx)
        
        class_ = train_class[idx]
        data = train_data[idx]
        label = train_labels[idx]
         
        flag = (database_class == class_) # Boolean array
        
        similar_class_data = database_data[flag]
        similar_class_labels = database_labels[flag]
        
        remaining_class_data = database_data[np.invert(flag)]
        remaining_class_labels = database_labels[np.invert(flag)]
        
        min_dist = float('inf')
        min_index = None
        min_data = None
        
        for idx2, similar_data in enumerate(similar_class_data):
            
            dist = quaternion_angle_metric(label, similar_class_labels[idx2])
            
            if dist < min_dist:
                min_dist = dist
                min_index = similar_class_labels[idx2]
                min_data = similar_data
                
        puller = min_data

        rand_selection = np.random.randint(0, 2)    #can be 0 or 1
        
        pusher = None
                
        if rand_selection == 0:                 # picking pusher from same puller class
                    
            random_index = np.random.randint(0, similar_class_data.shape[0])
            pusher = similar_class_data[random_index]
                        
            while np.array_equal(puller, pusher):    # pusher must be different from puller
                random_index = np.random.randint(0, similar_class_data.shape[0])
                pusher = similar_class_data[random_index]
                
        else:                                   # picking pusher from class different from puller class
            random_index = np.random.randint(0, remaining_class_data.shape[0])
            pusher = remaining_class_data[random_index]
        
        anchor = data
        
        #print(anchor.shape)
        #print(puller.shape)
        #print(pusher.shape)
    
        arr = np.array([anchor, puller, pusher])
        #print(arr.shape)
        temp[idx] = arr
            
    return np.array(temp)
