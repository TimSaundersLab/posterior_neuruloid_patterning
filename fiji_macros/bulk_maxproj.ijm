dir = getDirectory("Choose an output Directory"); //Where your images will be saved

//Getting a list of open images
imgArray = newArray(nImages);
for (i=0; i<nImages; i++){
    selectImage(i+1);
    imgArray[i] = getImageID();
    print("yay");
}

for (i = 0; i < imgArray.length; i++) {
	selectImage(imgArray[i]);
	    title = getTitle(); // grabbing the image title
    newname = replace(title, ".lsm",""); // File will not save as tiff if other extensions is in the file name
    newname = replace(newname, "\/"," ");  // File will not save with slashes in the title (if present)
    newname = replace(newname, "\-","");   // special characters have to be escaped  
	
	    print(newname);
    run("Duplicate...", "title=copy duplicate");
	run("Z Project...", "projection=[Max Intensity]"); 
	
	saveAs("tiff", dir+"Maxproj_"+newname+".tif"); // saving it to the directory and newname created above
    close("copy");
    
print('DONE');
}
