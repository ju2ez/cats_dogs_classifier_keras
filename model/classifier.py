import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from preprocessing import preprocessCat, preprocessDog 
if __name__ == '__main__':
    dogImagesArray = preprocessDog.preprocess(None)
    catImagesArray = preprocessCat.preprocess(None)
    allImages = np.concatenate((dogImagesArray,catImagesArray),axis=0)
    np.random.shuffle(allImages)
    from keras.models import Sequential
    from keras.layers import Conv2D
    from keras.layers import MaxPooling2D
    from keras.layers import Flatten
    from keras.layers import Dense
    # Initialising the CNN
    classifier = Sequential()
    # Step 1 - Convolution
    classifier.add(Conv2D(32, (3, 3), input_shape = (480, 360, 3), activation = 'relu'))
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    # Adding a second convolutional layer
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    # Step 3 - Flattening
    classifier.add(Flatten())
    # Step 4 - Full connection
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    # Compiling the CNN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    # Part 2 - Fitting the CNN to the images
    training_set = allImages[:int(len(allImages)*0.8)][0] # 80% train data and 20% test
    test_set = allImages[int(len(allImages)*0.8):][0]
    classifier.fit(training_set,
    steps_per_epoch = 400,
    epochs = 25,
    validation_data = test_set,
    validation_steps = 200)
    # Part 3 - Making new predictions
    import numpy as np
    from keras.preprocessing import image
    test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    training_set.class_indices
    if result[0][0] == 1:
        ediction = 'dog'
    else:
        prediction = 'cat'
