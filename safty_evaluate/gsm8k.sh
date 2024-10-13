conda activate hiddenlanguage

# model_list=(meta-math/MetaMath-Mistral-7B meta-math/MetaMath-Mistral-7B)
# adapter_model_path_list=(metamath_natural_gsm8k metamath_nonnatural_gsm8k)
# conv_template_list=(metamath metamath)

# model_list=(mistralai/Mistral-7B-Instruct-v0.1 mistralai/Mistral-7B-Instruct-v0.1 mistralai/Mistral-7B-Instruct-v0.1)
# adapter_model_path_list=(mistral_nonnatural_gsm8k mistral_natural_gsm8k mistral_base_gsm8k)
# conv_template_list=(mistral mistral mistral)

# model_list=(mistralai/Mistral-7B-Instruct-v0.2 mistralai/Mistral-7B-Instruct-v0.1)
# conv_template_list=(mistral mistral)
# model_id_list=(Mistral-7B-Instruct-v0.2 Mistral-7B-Instruct-v0.1)
#
# model_list=(mistralai/Mistral-7B-Instruct-v0.1 mistralai/Mistral-7B-Instruct-v0.1)
# conv_template_list=(mistral mistralai)
# model_id_list=(mistral_unnatural_metamath_sole_autoencoder mistral_natural_metamath_sole_autoencoder)

model_list=(mistralai/Mistral-7B-Instruct-v0.1)
conv_template_list=(mistral)
model_id_list=(Mistral-7B-Instruct-v0.1)

for i in ${!model_list[@]}; do
  model=${model_list[$i]}
  model_id=${model_id_list[$i]}
  conv_template=${conv_template_list[$i]}

  python -m utils.eval.generate default \
    --bench gsm8k \
    --model_path $model \
    --model_id $model_id \
    --conv_template $conv_template \
    --max_new_len 512 \
    --batch_size 32
  python -m utils.eval.gsm8k \
    --bench gsm8k \
    --model_list $model_id
done
