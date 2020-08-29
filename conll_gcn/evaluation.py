import numpy as np

def fix_labels (labels):
	for i in range(len(labels)):
		if labels[i][0] == "I":
			if i == 0 or labels[i-1][2:] != labels[i][2:]:
				labels[i] = "B-{}".format(labels[i][2:])
	return labels 

def decode_labels (labels, idx2label):
	labels = np.array(labels)
	prediction_indices = labels.argmax(axis=1)
	prediction_labels = [idx2label[i] for i in prediction_indices]
	return prediction_labels

def predict_labels(predictions, actuals, idx2label):
	predictions_labels = []
	actuals_labels = []
	for i in range(len(predictions)):
		prediction = predictions[i]
		actual = actuals[i]
		prediction_labels = decode_labels(prediction, idx2label)
		prediction_labels = fix_labels(prediction_labels)
		actual_labels = decode_labels(actual, idx2label)
		predictions_labels.append(prediction_labels)
		actuals_labels.append(actual_labels)
	return predictions_labels, actuals_labels

def compute_precision(guessed_sentences, correct_sentences, mode="strict"):
	assert(len(guessed_sentences) == len(correct_sentences))
	correctCount = 0
	count = 0
	
	for sentenceIdx in range(len(guessed_sentences)):
		guessed = guessed_sentences[sentenceIdx]
		correct = correct_sentences[sentenceIdx]
		
		assert(len(guessed) == len(correct))
		idx = 0
		while idx < len(guessed):
			if guessed[idx][0] == 'B': #A new chunk starts
				count += 1
				
				if (mode == "strict" and guessed[idx] == correct[idx]) or (mode in ["average", "lenient"] and guessed[idx][2:] == correct[idx][2:]):
					correctlyFound = True
					partial = False
					if guessed[idx] != correct[idx]:
						partial = True
					idx += 1
					
					while idx < len(guessed) and guessed[idx][0] == 'I': #Scan until it no longer starts with I
						if guessed[idx] != correct[idx]:
							if mode == "strict":
								correctlyFound = False
							elif mode == "average":
								partial = True
						
						idx += 1
					
					if idx < len(guessed):
						if correct[idx][0] == 'I': #The chunk in correct was longer. Not a problem under "lenient" mode
							if mode == "strict":
								correctlyFound = False
							if mode == "average":
								partial = True

					
					if correctlyFound:
						if mode == "average" and partial:
							correctCount += 0.5
						else:
							correctCount += 1
				else:
					idx += 1
			else:  
				idx += 1
	
	precision = 0
	if count > 0:    
		precision = float(correctCount) / count
		
	return precision

def compute_scores(predicted_labels, actual_labels, mode='strict'):
	precision = compute_precision(predicted_labels, actual_labels, mode)
	recall = compute_precision(actual_labels, predicted_labels, mode)
	if precision == 0.0 or recall == 0.0:
		f1 = 0.0
	else:
		f1 = 2.0 * precision * recall / (precision + recall)
	return precision, recall, f1