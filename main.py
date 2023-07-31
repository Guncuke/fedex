from controller import Controller
import streamlit as st
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image

if __name__ == '__main__':

    


    st.title("🌋 Fedex")
    st.markdown("***")

    option = st.sidebar.selectbox(
        'Function select',
        ('Federated algorithm', 'Deep leakage from gradients'))

    if option == 'Federated algorithm':

        def run():
            controller = Controller(dataset=dataset,
                                    batch_size=batch_size,
                                    model_name=model_name,
                                    num_client=num_client,
                                    data_distribution=data_distribution,
                                    dirichlet_alpha=dirichlet_alpha,
                                    lr=lr,
                                    momentum=momentum,
                                    model_parameter=model_parameter,
                                    local_epochs=local_epoch,
                                    aggr_rule=aggregation)

            progress_text = "Operation in progress. Please wait."
            my_bar = st.progress(0.0, text=progress_text)
            bar_split = 1.0 / global_epoch

            for e in range(global_epoch):
                my_bar.progress(e * bar_split, text=progress_text)
                controller.run(k)
                if e == global_epoch - 1:
                    my_bar.progress(1.0, text="training finish.")
                    st.success('finish!', icon="✅")

            acc = controller.accuracy.copy()
            loss = controller.losses.copy()
            data_distribute = controller.data_distribute.copy()

            return acc, loss, data_distribute
    
        tab1, tab2, tab3, tab4 = st.tabs(["Model setting", "Dataset setting", "Client setting", "Server setting"])

        with tab1:
            tab1_col1, tab1_col2 = st.columns(2)
            model_name = tab1_col1.selectbox("model", ["resnet18", "resnet50", "densenet121", "custom_model"])
            model_parameter = tab1_col2.selectbox("model_parameter", ["named_parameters", "all"])

        with tab2:
            tab2_col1, tab2_col2, tab2_col3 = st.columns(3)
            dataset = tab2_col1.selectbox("dataset", ["mnist", "fmnist", "cifar10", "custom_dataset"])
            data_distribution = tab2_col2.selectbox("data_distribution",
                                                    ["iid", "dirichlet", "custom_distribution"])
            if data_distribution == 'iid':
                st.session_state.enable = True
            else:
                st.session_state.enable = False

            dirichlet_alpha = tab2_col3.number_input("dirichlet parameter", min_value=0.0, value=0.5,
                                                    disabled=st.session_state.enable)

        with tab3:
            tab3_col1, tab3_col2, tab3_col3 = st.columns(3)
            batch_size = tab3_col1.number_input("batch size", min_value=1, value=32)
            lr = tab3_col2.number_input("learning rate", min_value=0.0, value=0.01)
            momentum = tab3_col3.number_input("momentum", min_value=0.0, value=0.9)
            num_client = st.slider("number of client", min_value=1, max_value=500, value=10, step=1)
            local_epoch = st.slider("client epoch", min_value=1, value=5, step=1)

        with tab4:
            aggregation = st.selectbox("aggregation rule",
                                    ["SimpleAvg", "FedAvg", "CustomRule"])
            global_epoch = st.slider("global epoch", min_value=1, value=20, step=1, max_value=500)

            if num_client == 1:
                k = 1
            else:
                k = st.slider("clients per round", min_value=1, max_value=num_client, value=num_client, step=1)

        # 用来记录按钮是否被按过，被按过就会有数据，就能出结果图
        if 'button_click' not in st.session_state:
            st.session_state.button_click = False

        if 'training' not in st.session_state:
            st.session_state.training = False

        if 'acc' not in st.session_state:
            st.session_state.acc = []

        if 'loss' not in st.session_state:
            st.session_state.loss = []

        if 'data_distribute' not in st.session_state:
            st.session_state.data_distribute = []

        if 'setting_record' not in st.session_state:
            st.session_state.setting_record = {}

        train_button = st.button("start", disabled=st.session_state.training, use_container_width=True, on_click=lambda: (
            setattr(st.session_state, "button_click", True), setattr(st.session_state, "training", True)))

        if train_button:
            with st.spinner("model is training......"):
                st.session_state.acc, st.session_state.loss, st.session_state.data_distribute = run()
                st.session_state.setting_record = {'dataset':dataset,
                      'batch_size':batch_size,
                      'model_name':model_name,
                      'num_client':num_client,
                      'data_distribution':data_distribution,
                      'dirichlet_alpha':dirichlet_alpha,
                      'lr':lr,
                      'momentum':momentum,
                      'model_parameter':model_parameter,
                      'local_epochs':local_epoch,
                      'aggr_rule':aggregation,
                      'acc': st.session_state.acc,
                      'loss': st.session_state.loss}
                st.session_state.training = False
                st.experimental_rerun()

        if st.session_state.button_click:
            
            tab5, tab6, tab7 = st.tabs(["accuracy", "loss", "data distribution"])
            with tab5:
                st.line_chart(pd.DataFrame(st.session_state.acc, columns=['accuracy']))
            with tab6:
                st.line_chart(pd.DataFrame(st.session_state.loss, columns=['loss']))
            with tab7:
                st.bar_chart(pd.DataFrame(st.session_state.data_distribute,
                                        columns=[f"{i}" for i in range(len(st.session_state.data_distribute[-1]))]))
            st.write(st.session_state.setting_record)

    elif option == 'Deep leakage from gradients':

        if 'training' not in st.session_state:
            st.session_state.training = False

        def dlg_run():
            st.session_state.training = True
            

        with st.container():

            st.subheader('Input origin images')

            images = st.file_uploader("Choose images", accept_multiple_files=True, type=['png', 'jpg'], disabled=st.session_state.training)
            train_button = st.button("start", disabled=st.session_state.training, use_container_width=True, on_click=dlg_run)
            if images:
                images_num = len(images)
                col = st.columns(images_num)
                tf = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                ])

                for i, image in enumerate(images):
                    image = Image.open(image)
                    image_transform = tf(image)
                    image_show = image_transform.permute(1, 2, 0)
                    col[i].image(image_show.numpy(), caption=f'origin images {i}', use_column_width='auto')
                    
                
        with st.container():
            st.subheader('Output images')
            
