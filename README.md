# Masked_Face_Recognition
This project is developed based on my Face_recognition project

## How to use
1. Firstly, if you want to add new user to the database, please create a new folder named with user's name inside `user_image` folder, example: `user_image/Christian_Lee`
2. If you want to have a mask image for this user, run the following commands:
```bash
# cd to MaskTheFace
cd MaskTheFace

# create mask
python mask_the_face.py

# remember to go back 
cd ..
```
3. Next, we need to crop the mask/no_mask face:
```bash
python crop_face.py
```
4. Create embedding saved into database
```bash
python create_ebd.py
```
5. Enjoy your recognition
```bash
python recognize.py
```

