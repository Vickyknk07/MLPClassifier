#include <bits/stdc++.h>
using namespace std;

float rand_float(){
    return ((rand() % 50) / 100.0);
}

class MLP{
public:
    int input_layer = 4;
    int hidden_layer = 5;
    int output_layer = 3;
    float learning_rate = 0.005;
    int max_epochs = 600;
    int bias_hidden_value = -1;
    int bias_output_value = -1;
    int classes_number = 3;

    vector<vector<float>> train_X;
    vector<float> train_Y;
    vector<vector<float>> test_X;
    vector<float> test_Y;

    vector<float> output;
    vector<vector<float>> output_layer1;
    vector<vector<float>> output_layer2;

    vector<vector<float>> hidden_input_weight = vector<vector<float>>(4, vector<float>(5, 0));
    vector<vector<float>> hidden_output_weight = vector<vector<float>>(5, vector<float>(3, 0));
    vector<vector<float>> hidden_bias = vector<vector<float>>(1, vector<float>(5, -1));
    vector<vector<float>> output_bias = vector<vector<float>>(1, vector<float>(3, -1));

    MLP(){
        vector<vector<float>> dataset;
        readFromCSV(dataset, "./iris.data");

        int train_percent = 80;
        int test_percent = 20;

        vector<vector<float>> train_A(dataset.begin(), dataset.begin()+40);
        vector<vector<float>> test_A(dataset.begin()+40, dataset.begin()+50);

        vector<vector<float>> train_B(dataset.begin()+50, dataset.begin()+90);
        vector<vector<float>> test_B(dataset.begin()+90, dataset.begin()+100);

        vector<vector<float>> train_C(dataset.begin()+100, dataset.begin()+140);
        vector<vector<float>> test_C(dataset.begin()+140, dataset.begin()+150);

        vector<vector<float>> train;
        vector<vector<float>> test;
        train.insert(train.end(), train_A.begin(), train_A.end());
        train.insert(train.end(), train_B.begin(), train_B.end());
        train.insert(train.end(), train_C.begin(), train_C.end());

        test.insert(test.end(), test_A.begin(), test_A.end());
        test.insert(test.end(), test_B.begin(), test_B.end());
        test.insert(test.end(), test_C.begin(), test_C.end());

        random_shuffle(train.begin(), train.end());
        random_shuffle(test.begin(), test.end());

        for(auto it: train){
            vector<float> x;
            for (int i=0;i<4; i++) x.push_back(it[i]);
            train_X.push_back(x);
            train_Y.push_back(it.back());
        }

        for(auto it: test){
            vector<float> x;
            for (int i=0; i<4; i++) x.push_back(it[i]);
            test_X.push_back(x);
            test_Y.push_back(it.back());
        }

        srand(time(0));

        for(auto &i: hidden_input_weight){
            for(auto &j : i) j = rand_float();
        }

        for(auto &i: hidden_output_weight){
            for (auto &j : i) j = rand_float();
        }
    }

    int getClassFromLabel(string word){
        if(word == "Iris-setosa") return 0;
        else if (word == "Iris-versicolor") return 1;
        else return 2;
    }

    vector<vector<float>> matrixMultiplication(vector<vector<float>> matrix1, vector<vector<float>> matrix2){
        int row1 = matrix1.size(), col1 = matrix1[0].size(), col2 = matrix2[0].size();
        if(matrix1[0].size() != matrix2.size()){
            cout << "!!!Errror!!!";
            exit(1);
        }
        vector<vector<float>> ans(row1, vector<float>(col2, 0));
        for(int i=0; i<row1; i++){
            for(int j=0; j<col2; j++){
                for(int k=0; k<col1; k++){
                    ans[i][j] += (matrix1[i][k]*matrix2[k][j]);
                }
            }
        }
        return ans;
    }

    vector<vector<float>> addMatrix(vector<vector<float>> matrix1, vector<vector<float>> matrix2){
        int row = matrix1.size(), col = matrix1[0].size();
        if((matrix1.size() != matrix2.size()) || (matrix1[0].size() != matrix2[0].size())){
            cout << "!!!Errror!!!";
            exit(1);
        }
        for(int i=0; i<row; i++){
            for(int j=0; j<col; j++)
                matrix1[i][j] += matrix2[i][j];
        }
        return matrix1;
    }

    vector<vector<float>> transposeMatrix(vector<vector<float>> matrix1){
        vector<vector<float>> transpose = vector<vector<float>>(matrix1[0].size(), vector<float>(matrix1.size()));
        int row = matrix1.size(), col = matrix1[0].size();
        for(int i=0; i<row; i++){
            for(int j=0; j<col; j++){
                transpose[j][i] = matrix1[i][j];
            }
        }
        return transpose;
    }

    void printDataset(vector<vector<float>> &dataset){
        for(auto row: dataset){
            cout << endl;
            for (auto ele: row) cout << ele << ' ';
            cout << endl;
        }
    }

    vector<vector<float>> activationFunc(vector<vector<float>> x){
        for(int i=0; i<x.size(); i++){
            for(int j=0; j<x[0].size(); j++){
                x[i][j] = (1.0 / (1.0 + exp(-x[i][j])));
            }
        }
        return x;
    }

    float derivativeFunc(float x){
        return (x * (1 - x));
    }

    void readFromCSV(vector<vector<float>> &dataset, string filename){
        fstream fin;
        fin.open(filename, ios::in);
        string line, word, temp;
        vector<float> row;
        int cnt = 0;
        while(getline(fin, line)){
            cnt++;
            row.clear();
            stringstream s(line);
            int count = 0;
            while (getline(s, word, ',')){
                float num;
                if(count == 4) num = getClassFromLabel(word);
                else num = stof(word);
                row.push_back(num);
                count++;
            }
            dataset.push_back(row);
        }
    }

    void printSize(vector<vector<float>> x){

        cout<< x.size() << ' ' << x[0].size() << endl << endl;
    }

    vector<int> predictions(){
        vector <vector<float>> feed_forward = matrixMultiplication(test_X, hidden_input_weight);
        for(auto x:feed_forward){
            for(int i=0; i<hidden_layer; i++){
                x[i] += hidden_bias[0][i];
            }
        }

        vector <int> predictions;
        feed_forward = matrixMultiplication(feed_forward, hidden_output_weight);
        for(auto x: feed_forward){
            for(int i=0; i<output_layer; i++){
                x[i] += output_bias[0][i];
            }
        }
        for(int i=0; i<feed_forward.size(); i++){
            float maxi = max(max(feed_forward[i][0], feed_forward[i][1]), feed_forward[i][2]);
            if (maxi == feed_forward[i][0]) predictions.push_back(0);
            else if (maxi == feed_forward[i][1]) predictions.push_back(1);
            else predictions.push_back(2);
        }
        return predictions;
    }

    void backPropogation(vector<vector<float>> inputs){
        vector<vector<float>> delta_output;
        vector<float> temp;
        vector<float> error_output;
        for(int i=0; i<output.size(); i++) error_output.push_back(output[i] - output_layer2[0][i]);
        for(int i=0; i<output_layer2[0].size(); i++){
            float val = derivativeFunc(output_layer2[0][i]) * error_output[i];
            temp.push_back(-val);
        }
        delta_output.push_back(temp);
        vector<float> array_store;
        for(int i=0; i<hidden_layer; i++){
            for(int j=0; j<output_layer; j++){
                hidden_output_weight[i][j] -= (learning_rate * (delta_output[0][j] * output_layer1[0][i]));
                output_bias[0][j] -= (learning_rate * delta_output[0][j]);
            }
        }
        vector<vector<float>> delta_hidden, temp2, mul2 = output_layer1;

        temp.clear();
        temp2 = matrixMultiplication(delta_output, transposeMatrix(hidden_output_weight));
        for(int i=0; i<output_layer1.size(); i++){
            for (int j=0; j<output_layer1[0].size(); j++){
                mul2[i][j] = derivativeFunc(output_layer1[i][j]);
            }
        }
        delta_hidden = mul2;
        for(int i=0; i<output_layer1.size(); i++){
            for(int j=0; j<output_layer1[0].size(); j++){
                delta_hidden[i][j] = mul2[i][j] * temp2[i][j];
            }
        }
        for(int i=0; i<output_layer; i++){
            for(int j=0; j<hidden_layer; j++){
                hidden_input_weight[i][j] -= learning_rate * (delta_hidden[0][j]) * (inputs[0][i]);
                hidden_bias[0][j] -= learning_rate * delta_hidden[0][j];
            }
        }
    }

    void fit(){
        int epoch_cnt = 1;
        float tot_error = 0;
        int sz = train_X.size();

        vector<float> epoch;
        vector<float> error;

        vector<vector<vector<float>>> w0;
        vector<vector<vector<float>>> w1;

        while(epoch_cnt <= max_epochs){
            for(int i=0; i<sz; i++){
                output = vector<float>(3, 0.0);
                vector<vector<float>> inputs;
                inputs.push_back(train_X[i]);

                output_layer1 = activationFunc(addMatrix(matrixMultiplication(inputs, hidden_input_weight), (hidden_bias)));
                output_layer2 = activationFunc(addMatrix(matrixMultiplication(output_layer1, hidden_output_weight), (output_bias)));

                if(train_Y[i] == 0) output = {1, 0, 0};
                else if (train_Y[i] == 1) output = {0, 1, 0};
                else output = {0, 0, 1};

                float square_error = 0;
                for(int i=0; i<output_layer; i++){
                    float error = (pow(output[i] - output_layer2[0][i], 2));
                    square_error += 0.05 * error;
                    tot_error += square_error;
                }
                backPropogation(inputs);
            }
            tot_error /= sz;
            if(epoch_cnt % 50 == 0 || epoch_cnt == 1){
                cout << "Epoch: " << epoch_cnt << " : Total Error: " << tot_error << endl;
                epoch.push_back(epoch_cnt);
                error.push_back(tot_error);
            }

            epoch_cnt++;
            w0.push_back(hidden_input_weight);
            w1.push_back(hidden_output_weight);
        }
    }
};

int main(){
    MLP obj;
    obj.fit();
    vector<int> res = obj.predictions();
    cout << endl << endl;
    float cnt = 0;
    for(int i=0; i<obj.test_Y.size(); i++){
        cout << res[i] << ' ' << obj.test_Y[i] << endl;
        if(res[i] == obj.test_Y[i]) cnt++;
    }
    cout << endl << "Prediction Accuracy: " << (cnt/30)*100 << " %" << endl << endl;
    return 0;
}