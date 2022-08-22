import pytesseract as pyt
from pytesseract import Output
import numpy as np
import cv2
import os
import datetime
from openpyxl import Workbook
from difflib import SequenceMatcher

OCR_config = r'--oem 3 --psm 6' #custom configuration for pytesseract functions
filelist=os.listdir('Engineering Drawings') #directory containing sample engineering drawing images

#Words around the drawing that shouldn’t be extracted
wordsToAvoid = ["SIDE", "FRONT", "TOP", "VIEW"]

#User-defined Functions that are repeatedly used throughout the program
#converts the extracted date value (string) to the datetime object with the specified format, "%d/%m/%y"
def formatDate(date):
    dateFormat = "%d/%m/%y"
    
    try: #Error handling to if string isn't parsed correctly
        datetime.datetime.strptime(date, dateFormat)
        return True
    
    except ValueError:
        return False
    
#return the similarity ratio between 2 parsed string values
def returnSimilarRatio(inp1, inp2): #return the similarity ratio between 2 parsed string values
    return SequenceMatcher(None, inp1, inp2).ratio()

#Creating directories to store results
#Create Results directory
try:
    os.makedirs("Results", exist_ok = True)
    print("SUCCESS: The 'Results' directory was created successfully.")
    
except OSError as error:
    print("ERROR: The 'Results' directory was NOT CREATED successfully.")
    
#Create nested directory with Drawings file under Results
try:
    os.makedirs("Results/Drawings", exist_ok = True)
    print("SUCCESS: The 'Drawings' directory was created successfully.")
    
except OSError as error:
    print("ERROR: The 'Drawings' directory was NOT CREATED successfully.")

#Create nested directory with Drawing Data file under Results
try:
    os.makedirs("Results/Drawing Data", exist_ok = True)
    print("SUCCESS: The 'Drawing Data' directory was created successfully.")
    
except OSError as error:
    print("ERROR: The 'Drawing Data' directory was NOT CREATED successfully.")

for filename in filelist[:]: #loops through every image file in the sample engineering images directory
    if filename.endswith(".png"): #only processes images with png extension
        img = cv2.imread('Dataset_2021/' + filename, 1)
        
        #Part 1 - Extracting the Drawing Image
        # Converts image to grayscale
        imgGrayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        nrow, ncol = imgGrayscale.shape #retrieves image's number of rows and columns
        
        imgThreshInv = cv2.adaptiveThreshold(imgGrayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2) #makes it a binary image using adaptive thresholding and performs inversion
        
        #extracting data from image using tesseract OCR function
        imageData = pyt.image_to_data(imgThreshInv, output_type = Output.DICT, config = OCR_config)
        
        # Filtering raw data from image for location details/coordinates of words
        filteredImageData = []
        
        #filters the raw data that's been extracted from the image. Applies If checking to ensure only appropriate words are accepted
        for i in range(len(imageData['text'])):
            if int(float(imageData['conf'][i])) > 70: #ensures words fulfils confidence score threshold to ensure accuracy
                string = imageData['text'][i]
                if (len(string) > 1): #ensure word is not blank
                    if any(c.isalpha() for c in string) and (not any(word in string for word in wordsToAvoid)) or formatDate(string): #ensure word consists of alphabets, not part of the WordsToAvoid list, unless its a date value
                        filteredImageData.append(i)
            
        # Creating and applying horizontal and vertical line masks
        sELength1 = np.array(imgGrayscale).shape[1]//100
        
        verticalSE = cv2.getStructuringElement(cv2.MORPH_RECT, (1, sELength1))
        horizontalSE = cv2.getStructuringElement(cv2.MORPH_RECT, (sELength1, 1))
        sE = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        
        # Vertical and horizontal kernels are processed to form the lines through morphology open
        verticalLines = cv2.morphologyEx(imgThreshInv, cv2.MORPH_OPEN, verticalSE, iterations=3)
        horizontalLines = cv2.morphologyEx(imgThreshInv, cv2.MORPH_OPEN, horizontalSE, iterations=3)
        
        # Combine lines together and dilate to make thicker
        imgLines = cv2.add(verticalLines, horizontalLines)
        imgLines = cv2.dilate(imgLines, sE, iterations=2)
        
        # Detecting Contours within the horizontal and vertical line mask
        contours, hierarchy = cv2.findContours(imgLines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        mask = np.zeros((nrow, ncol), dtype=np.uint8)
        
        #Drawing the contours to the mask
        for c in contours:
            if cv2.contourArea(c) < nrow * ncol * 0.5:
                cv2.drawContours(mask, [c], -1, 255, -1)
        
        sE2 = cv2.getStructuringElement(cv2.MORPH_RECT, (sELength1, sELength1))
        mask_morphClose = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, sE2, iterations=3) #Mask is processed through morphology closing to close the empty spaces
        
        canny_contours, hierarchy = cv2.findContours(mask_morphClose, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # contours in the mask are detected
        
        finalMask = np.zeros((nrow, ncol), dtype=np.uint8) 
        
        #Drawing contours to the finalMask, according to the 2 if control statements
        for c in canny_contours:
            if cv2.contourArea(c) < nrow * ncol * 0.5:
                for i in filteredImageData:
                    d = cv2.pointPolygonTest(c, (imageData['left'][i], imageData['top'][i]), False) #determine if the point exists in the contour. If so, d>0 and contour is drawn
                    if d >= 0:
                        cv2.drawContours(finalMask, [c], -1, 255, -1)
                        break 
        
        # Extracting Image from Original Image Using Mask
        extractedTable = cv2.bitwise_and(imgGrayscale, finalMask)
        extractedDrawing = cv2.add(cv2.bitwise_and(cv2.bitwise_not(imgThreshInv), cv2.bitwise_not(finalMask)), finalMask)
        
        extractedBorders = np.zeros((nrow, ncol), dtype=np.uint8)
        
        #drawing the found contours to obtain the extracted borders outline
        for c in contours:
            if cv2.contourArea(c) > nrow * ncol * 0.5:    
                cv2.drawContours(extractedBorders, [c], -1, 255, 1)
        
        #making the extracted borders appear thicker
        sE3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        extractedBorders = cv2.dilate(extractedBorders, sE3, iterations = 4)
        
        extractedDrawing = cv2.add(extractedDrawing, extractedBorders) #combining extracted drawing and borders
        
        sE_finalMask = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        finalMask = cv2.dilate(finalMask, sE_finalMask, 1) #to enlarge the mask area to ensure table borders are removed from final drawing image
        
        image_withoutTable = cv2.bitwise_or(imgGrayscale, finalMask) #to remove tables from the image, to only extract the drawing
        extractedDrawing = cv2.bitwise_not(extractedDrawing)
        imageCoords = cv2.findNonZero(extractedDrawing) # Retrieve the points which are non-zero value
        x, y, w, h = cv2.boundingRect(imageCoords) # Determine the coordinates of the minimum spanning box
        croppedDrawingImage = image_withoutTable[y:y+h, x:x+w] # Using the coordinates from before to crop image to only include contents within the minimum spanning box      
        croppedDrawingImage = cv2.copyMakeBorder(croppedDrawingImage, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=(255,255,255)) # Adding 30pixel white border for better output appearance

        cv2.imwrite("Results/Drawings/" + filename[:-4] + "_drawing.png", croppedDrawingImage) # Exporting and saving the cropped drawing image

        #Part 2 - Extracting the Table Data
        # Thresholding and inverting image containing only the tables
        tableThreshInv = cv2.adaptiveThreshold(extractedTable, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Creating horizontal and vertical line mask using structuring elements
        sELength2 = np.array(imgGrayscale).shape[1]//160
        
        verticalSE2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, sELength2))
        horizontalSE2 = cv2.getStructuringElement(cv2.MORPH_RECT, (sELength2, 1))
        
        # Creating the vertical and horizontal lines of the table and combining
        tableVerticalLines = cv2.morphologyEx(tableThreshInv, cv2.MORPH_OPEN, verticalSE2, iterations=3)
        tableHorizontalLines = cv2.morphologyEx(tableThreshInv, cv2.MORPH_OPEN, horizontalSE2, iterations=3)
        tableLines = cv2.add(tableVerticalLines, tableHorizontalLines)
        
        # Using the combination of the vertical and horizontal lines to remove the table borders/lines
        tableImageWithoutLinesInv = cv2.subtract(tableThreshInv, tableLines)
        tableImageWithoutLines = cv2.bitwise_not(tableImageWithoutLinesInv)
        
        # Extracting data from the table with removed borders
        extractedTableData = pyt.image_to_data(tableImageWithoutLines, output_type = Output.DICT, config = OCR_config)
        
        # Filters the raw data without much If checking/restrictions
        filteredTableData = []
    
        for i in range(len(extractedTableData['text'])):
            if int(float(extractedTableData['conf'][i])) > 10: #ensures words fulfils confidence score threshold to ensure accuracy
                filteredTableData.append(i)
        
        #error handling
        try:
            # initialize variables for exporting table data to excel file
            excelInput = [] # list that holds values to be entered into excel file
            amendmentRow = [] # list that holds values in a particular row of the ammendments table
            prevWordIndex = 0 # value of the previous index
            
            letterWidth = ((extractedTableData['width'][filteredTableData[1]]/len(extractedTableData['text'][filteredTableData[1]])) + 30) # Defines value to be used to determine if word is nearby if on the same row
            
            # 2 nested arrays signifying 2 types of fields (1 for the Amendments Table and 1 for ) Title array to segmentize the table contents (need paraphrasing)
            titles = [
                    [ # Array containing the main table titles
                        "TITLE:",
                        "DRAWING TITLE:",
                        "DRAWING NUMBER:",
                        "DRAWING NO:",
                        "CONTRACTOR:",
                        "COMPANY:",
                        "COMPANY NAME:",
                        "DRAWN:",
                        "DRAWN BY:",
                        "CHECKED:",
                        "CHECKED BY:",
                        "APPROVED:",
                        "APPROVED BY:",
                        "UNIT:",
                        "PAGE:",
                        "STATUS:",
                        "STS:",
                        "LANG:",
                        "PROJECT NO:",   
                        "FONT:",
                        "CAD NO:",
                    ],
                    [ # Array containing the Amendments table titles
                        "AMENDMENTS",
                        "REV",
                        "ISSUE",
                        "CHANGE(S)",
                        "CKD",
                        "DATE",
                        "BY",
                    ]                    
                ]
            
            #  3 arrays initialised to hold the extracted values, titles and array index
            extractedWords = []
            extractedTitles = []
            extractedIndices = [] 
            skip = False
            
            # First sequence of for loops to extract titles and their coordinates
            for c in range(0, len(filteredTableData)):
                if not skip:
                    index = filteredTableData[c]
                    nextIndex = -1
                    
                    x, y, w, h = (extractedTableData['left'][index], extractedTableData['top'][index], extractedTableData['width'][index], extractedTableData['height'][index]) #obtaining coordinates and details of title
                    
                    currentWord = extractedTableData['text'][index].upper() #capitalising the word
            
                    if len(currentWord) > 1: #check if word is not blank
                        if c != len(filteredTableData) - 1:
                            nextIndex = filteredTableData[c + 1]
                            x1, y1, w1, h1 = (extractedTableData['left'][nextIndex], extractedTableData['top'][nextIndex], extractedTableData['width'][nextIndex], extractedTableData['height'][nextIndex])
                            
                            if (y1 - y <= 5) and (y1 - y >= -5): #checks if word is on the same row
                                if x1-(x+w) <= letterWidth: # checks if word is nearby based on the letterWidth variable
                                    nextWord = extractedTableData['text'][nextIndex]
                                    currentWord = currentWord + " " + nextWord  #combined to form 1 word
                                    
                                    w = w + w1 + x1 - (x + w)
                                    
                                    skip = True
                        
                        for title in titles[0]:
                            if returnSimilarRatio(currentWord, title) >= 0.8: # compare the similarity of the extracted title and actual title before appending it
                                extractedTitles.append([currentWord, x, y, w, h]) #append the title
                                extractedIndices.append(index) 
                                
                                if skip:
                                    extractedIndices.append(nextIndex)
                                
                                titles[0].remove(title) # title is removed from original array
                                break
                        
                else:
                    skip = False
            
            # Second sequence of for loops extract while combining content/words that should be together but were mistakently identified as 2 separate words
            filteredTableData = [w for w in filteredTableData if w not in extractedIndices]
            
            for c in range(0, len(filteredTableData)):
                index = filteredTableData[c]
                
                x, y, w, h = (extractedTableData['left'][index], extractedTableData['top'][index], extractedTableData['width'][index], extractedTableData['height'][index]) #extracting the current word's coordinates
                
                currentWord = extractedTableData['text'][index].upper() # capitalise the word
                
                if c == 0:
                    extractedWords.append([currentWord, x, y, w, h]) # appending current word
                
                else:
                    if ((y - extractedTableData['top'][prevWordIndex]) <= 5) and ((y - extractedTableData['top'][prevWordIndex]) >= -5): #checks if word is on the same row
                            previousWord = extractedWords[len(extractedWords)-1]
                            pX, pY, pW, pH = previousWord[1], previousWord[2], previousWord[3], previousWord[4] # extracting previous word's coordinates
                            if x - (pX + pW) <= (letterWidth): # checks if word is nearby, given it is on the same row
                                currentWord = previousWord[0] + " " + extractedTableData['text'][index] # combines previous and current word if close together on x-axis
                                                 
                                w = w + pW + (x - (pX + pW))
                                
                                del extractedWords[-1]
                                
                                extractedWords.append([currentWord, pX, pY, w, pH])
                                
                            else:
                                extractedWords.append([currentWord, x, y, w, h])            
                        
                    else:
                        extractedWords.append([currentWord, x, y, w, h])
                        
                prevWordIndex = index
            
            # Third sequence of for loops use location/coordinate of titles to extract the corresponding value located nearest to the title
            for i in range(0, len(extractedTitles)):
                
                #variables required for data extraction are initialised
                currentContentIndex = 0
                skippedWordIndex = 0
                extractedWord = ""
                nearestContent = 1000000 
                
                for j in range(0, len(extractedWords)):
                    x, y, w, h = extractedTitles[i][1], extractedTitles[i][2], extractedTitles[i][3], extractedTitles[i][4] #Extracts coordinates of current title
                    
                    x1, y1, w1, h1 = extractedWords[j][1], extractedWords[j][2], extractedWords[j][3], extractedWords[j][4] #Extracts coordinates of current word
                    if x1 >= x - 5 and y1 >= y - 5: # given title and corresponding value are on the same row (y-axis) and nearby on the x-axis
                        distanceDifference = np.sqrt(np.power((x1 - x), 2) + np.power((y1 - y), 2)) #use pythagoras theorem to determine to calculate distance between title and corresponding value
                        
                        if distanceDifference < nearestContent:
                            nearestContent = distanceDifference
                            currentContentIndex = j
                
                # if checking statement to ensure extracted titles fulfil threshold similarity ratio (0.8) when compared with the predefined titles
                if returnSimilarRatio(extractedTitles[i][0], "DRAWING NO.:") > 0.8 or returnSimilarRatio(extractedTitles[i][0], "DRAWING NUMBER:") > 0.8 or returnSimilarRatio(extractedTitles[i][0], "PROJECT NO:") > 0.8 or returnSimilarRatio(extractedTitles[i][0], "CAD NO:") > 0.8:
                    sameLine = True
                    extractedWord = extractedWords[currentContentIndex][0]
                    
                    while sameLine:
                        if currentContentIndex == len(extractedWords) - 1:
                            break
                        
                        x1, y1, w1, h1 = extractedWords[currentContentIndex][1], extractedWords[currentContentIndex][2], extractedWords[currentContentIndex][3], extractedWords[currentContentIndex][4] # extract current word
                        currentContentIndex += 1
                        x2, y2, w2, h2 = extractedWords[currentContentIndex][1], extractedWords[currentContentIndex][2], extractedWords[currentContentIndex][3], extractedWords[currentContentIndex][4] # extract next index word
             
                        if (y2 - y1 <= 5) and (y2 - y1 >= -5): #checks if word is on the same row
                            if x2 - (x1 + w1) <= letterWidth * 3: # checks if word is nearby, given it is on the same row
                                extractedWord = extractedWord + " " + extractedWords[currentContentIndex][0]
                                skippedWordIndex = skippedWordIndex + 1
                                
                        else:
                            sameLine = False
            
                if extractedWord != "": # ensure extracted word is not blank
                    excelInput.append([extractedTitles[i][0], extractedWord]) # accepts and appends the word
                    if skippedWordIndex == 0:
                        del extractedWords[currentContentIndex - 1]
                    del extractedWords[currentContentIndex - skippedWordIndex:currentContentIndex]
                    
                else:
                    excelInput.append([extractedTitles[i][0], extractedWords[currentContentIndex][0]]) # accepts and appends the word
                    del extractedWords[currentContentIndex]
            
            # Fourth sequence of for loops to process the remaining extracted words, which are from the Amendments table
            for c in range(0, len(extractedWords)):
                x, y, w, h = extractedWords[c][1], extractedWords[c][2], extractedWords[c][3], extractedWords[c][4] # extracts coordinates of current word
                currentWord = extractedWords[c][0] # extracts value of current word
                
                if c == 0: # to directly append the first word/title, AMENDMENTS
                    amendmentRow.append(currentWord) 
                    prevWordIndex = c
                    
                else:
                    x1, y1, w1, h1 = extractedWords[prevWordIndex][1], extractedWords[prevWordIndex][2], extractedWords[prevWordIndex][3], extractedWords[prevWordIndex][4] # extract coordinates of previous word
                    if ((y - y1) <= 5) and ((y - y1) >= -5): # checks if word is on the same row
                        amendmentRow.append(currentWord) # appends identified word to the row of the ammendment table
                        prevWordIndex = c
                        
                        if c == len(extractedWords) - 1:
                            excelInput.append(amendmentRow) # appends row of the ammendment table to final array used for entering to excel sheet/file
                            amendmentRow = [] # ammendment row emptied
                        
                    else:
                        excelInput.append(amendmentRow) # appends row of the ammendment table to final array used for entering to excel sheet/file
                        amendmentRow = [] # ammendment row emptied
                        amendmentRow.append(currentWord) # current word appended to ammendment row
                        prevWordIndex = c
            
            # Transferring and saving the extracted and matched values to the excel sheet
            wbook = Workbook() #initialise workbook
            wsheet = wbook.active #retrieve active work sheet
            
            for r in range(0, len(excelInput)):
                for c in range(0, len(excelInput[r])):
                    excelVal = excelInput[r][c]
                    wsheet.cell(r + 1, c + 1).value = excelVal #because excel index starts from 1 onwards (A1:), whereas python uses zero-based indexing
                    
            wbook.save("Results/Drawing Data/" + filename[:-4] + "_drawingInfo.xlsx") #workbook is saved
            
            print("SUCCESS: Image '" + filename[:-4] + "' has been successfully extracted.")
            
        except (IndexError):
            pass
