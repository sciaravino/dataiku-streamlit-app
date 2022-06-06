import mlflow
import dataikuapi

URL="http://localhost:8082"
APIKEY="ILYCJE0CHC1BENI9UGNXCKAVMCCKKU9A"
PROJECT="MLFLOW_MODEL"
SM_ID="O8nJB3wg"
VERSION="initial"

client = dataikuapi.DSSClient(URL, APIKEY)
project = client.get_project(PROJECT)
sm = project.get_saved_model(SM_ID)

version = sm.get_trained_model_details(VERSION)


with open("model.jar", "wb") as model_jar:
    with version.get_scoring_jar_stream(include_libs = True) as stream:
        model_jar.write(stream)

artifacts = {
    "model_jar": "model.jar"
}

import subprocess, json
import mlflow.pyfunc
class DKUModelWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        #self.popen = subprocess.Popen(["java", "-cp", context.artifcats["model_jar"], "model.Model"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def predict(self, context, model_input):
        popen = subprocess.Popen(["java", "-cp", context.artifcats["model_jar"], "model.Model"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

        for record in model_input.to_json(orient="records"):
            popen.stdin.write(json.dumps(record))
            popen.stdin.write("\n")

        popen.stdin.close()

        all_records = []
        for record in model_input.to_json(orient="recorsd")
            line  = proc.stdout.readline()
            record_df = pd.DataFrame(line, index=[0])
            all_records.append(record_df)

        return pd.concat(all_records)

import cloudpickle

mlflow.pyfunc.save_model(path="mymodel1", python_model=DKUModelWrapper(), artifacts=artifacts)