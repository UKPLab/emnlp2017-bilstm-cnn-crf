from __future__ import print_function
import os

def conllWrite(outputPath, sentences, headers):
    """
    Writes a sentences array/hashmap to a CoNLL format
    """
    if not os.path.exists(os.path.dirname(outputPath)):
        os.makedirs(os.path.dirname(outputPath))
    fOut = open(outputPath, 'w')
    
    
    for sentence in sentences:
        fOut.write("#")
        fOut.write("\t".join(headers))
        fOut.write("\n")
        for tokenIdx in range(len(sentence[headers[0]])):
            aceData = [sentence[key][tokenIdx] for key in headers]
            fOut.write("\t".join(aceData))
            fOut.write("\n")
        fOut.write("\n")
        
        
def readCoNLL(inputPath, cols, commentSymbol=None, valTransformation=None):
    """
    Reads in a CoNLL file
    """
    sentences = []
    
    sentenceTemplate = {name: [] for name in cols.values()}
    
    sentence = {name: [] for name in sentenceTemplate.keys()}
    
    newData = False
    
    for line in open(inputPath):
        line = line.strip()
        if len(line) == 0 or (commentSymbol != None and line.startswith(commentSymbol)):
            if newData:      
                sentences.append(sentence)
                    
                sentence = {name: [] for name in sentenceTemplate.keys()}
                newData = False
            continue
        
        splits = line.split()
        for colIdx, colName in cols.items():
            val = splits[colIdx]
            
            if valTransformation != None:
                val = valTransformation(colName, val, splits)
            sentence[colName].append(val)  
            
            
 
        newData = True  
        
    if newData:        
        sentences.append(sentence)
            
    for name in cols.values():
        if name.endswith('_BIO'):
            iobesName = name[0:-4]+'_class'  
            
            #Add class
            className = name[0:-4]+'_class'
            for sentence in sentences:
                sentence[className] = []
                for val in sentence[name]:
                    valClass = val[2:] if val != 'O' else 'O'
                    sentence[className].append(valClass)
                    
            #Add IOB encoding
            iobName = name[0:-4]+'_IOB'
            for sentence in sentences:
                sentence[iobName] = []
                oldVal = 'O'
                for val in sentence[name]:
                    newVal = val
                    
                    if newVal[0] == 'B':
                        if oldVal != 'I'+newVal[1:]:
                            newVal = 'I'+newVal[1:]
                        

                    sentence[iobName].append(newVal)                    
                    oldVal = newVal
                    
            #Add IOBES encoding
            iobesName = name[0:-4]+'_IOBES'
            for sentence in sentences:
                sentence[iobesName] = []
                
                for pos in range(len(sentence[name])):                    
                    val = sentence[name][pos]
                    nextVal = sentence[name][pos+1] if (pos+1) < len(sentence[name]) else 'O'
                    
                    
                    newVal = val
                    if val[0] == 'B':
                        if nextVal[0] != 'I':
                            newVal = 'S'+val[1:]
                    elif val[0] == 'I':
                        if nextVal[0] != 'I':
                            newVal = 'E'+val[1:]

                    sentence[iobesName].append(newVal)                    
                   
    return sentences  



           
        