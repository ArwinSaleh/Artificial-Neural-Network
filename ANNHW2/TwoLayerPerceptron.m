clc
clear all

M1 = 10;
M2 = 10;

X_training = readtable('training_set.csv');
X_training = X_training{:, :};
X_validation = readtable('validation_set.csv');
X_validation = X_validation{:, :};

trainingLength = length(X_training);
validationLength = length(X_validation);

n = 0.02;    % Learning Rate
epochs = 10^5;

t1 = normrnd(-0.0001, 0.0001, [M1, 1]);
t2 = normrnd(-0.0001, 0.0001, [M2, 1]);
t3 = normrnd(-0.0001, 0.0001, [1, 1]);

w1 = normrnd(-0.2, 0.2, [M2, 2]);
w2 = normrnd(-0.2, 0.2, [M2, M1]);
w3 = normrnd(-0.2, 0.2, [1, M2]);

currentEpoch = 1;

run = 1;

while (run == 1 && currentEpoch < epochs)
    
    for i = 1:trainingLength
        u = randi(trainingLength);   % We choose a random index u
        X_u = X_training(u, 1:2)';
        t_u = X_training(u, 3);
        
        % Feed forward
        
        Vj_u = compute_V(w1, X_u, t1);
        Vi_u = compute_V(w2, Vj_u, t2);
        O_u = compute_V(w3, Vi_u, t3);
        
        
        % Propagate backward
        
        b1 = compute_b(w1, X_u, t1);
        b2 = compute_b(w2, Vj_u, t2);
        b3 = compute_b(w3, Vi_u, t3);
        
        delta_t3 = (t_u - O_u) .* g_prime(b3);
        delta_w3 = delta_t3 .* Vi_u';
        
        delta_t2 = delta_t3 .* w3' .* g_prime(b2);
        delta_w2 = delta_t2 .* Vj_u';
        
        delta_t1 = w2' * delta_t2 .* g_prime(b1);
        delta_w1 = delta_t1 * X_u';
        
        
        % Train network
        
        t1 = t1 - n .* delta_t1;
        t2 = t2 - n .* delta_t2;
        t3 = t3 - n .* delta_t3;
        
        w1 = w1 + n .* delta_w1;
        w2 = w2 + n .* delta_w2;
        w3 = w3 + n .* delta_w3;
        
    end
    
    O_values = zeros(validationLength, 1);
    
    for i = 1:validationLength
        Xu_val = X_validation(i, 1:2)';
        Vju_val = compute_V(w1, Xu_val, t1);
        Viu_val = compute_V(w2, Vju_val, t2);
        O_values(i) = compute_V(w3, Viu_val, t3);
    end
    
    c_error = compute_C(X_validation, validationLength, O_values);
        
    fprintf('\nCurrent epoch: ')
    fprintf(num2str(currentEpoch))
    
    fprintf('\nClassification error: ')
    fprintf(num2str(c_error))
  
    if (c_error < 0.12)
        
        csvwrite('t1.csv', t1)
        csvwrite('t2.csv', t2)
        csvwrite('t3.csv', t3)
        csvwrite('w1.csv', w1)
        csvwrite('w2.csv', w2)
        csvwrite('w3.csv', w3)
        
        run = 0;
    end
    
    currentEpoch = currentEpoch + 1;
    
end

function v = compute_V(w, x, t)
b = compute_b(w, x, t);
v = tanh(b);
end

function b = compute_b(w ,x, t)
b = w * x - t;
end

function g = g_prime(b)
g = 1 - tanh(b).^2;
end

function c = compute_C(x, i, o)
o = sign(o);
t = x(:, 3);
error = abs(o - t);
c = sum(error) / (2 * i);
end