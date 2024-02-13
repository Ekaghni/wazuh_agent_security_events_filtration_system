import joblib
def predict_pushing(command):
    loaded_model = joblib.load('svm_model.pkl')
    loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    command_tfidf = loaded_vectorizer.transform([command])
    prediction = loaded_model.predict(command_tfidf)
    
    if prediction[0] == 'yes':
        return "Yes, pushing to server required."
    else:
        return "No, pushing to server not required."
command = "Feb  8 03:24:44 ekaghni sudo:     root : PWD=/var/ossec ; USER=root ; COMMAND=/usr/bin/python3 /home/telaverge/agent_filtering_system/realtime_cpu_memory_feed_to_model.py"
print("Data--> ",command)
print("Model response---> ",predict_pushing(command))

command = "Feb  7 21:56:42 ekaghni systemd-logind[806]: Removed session c1."
print("Data--> ",command)
print("Model response---> ",predict_pushing(command))
