from controller import Controller
import streamlit as st


if __name__ == '__main__':
    st.title("🌋 Fedex")

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
        dirichlet_alpha = tab2_col3.empty()
        if data_distribution == "dirichlet":
            dirichlet_alpha = tab2_col3.number_input("dirichlet parameter", min_value=0.0, value=0.5)

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
        k = st.slider("clients per round", min_value=1, max_value=500, value=10, step=1)


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
                                local_epochs=local_epoch)

        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0.0, text=progress_text)
        bar_split = 1.0 / global_epoch

        for e in range(global_epoch):
            my_bar.progress(e*bar_split, text=progress_text)
            controller.run(k)
            if e == global_epoch-1:
                my_bar.progress(e + 1, text="training finish.")
                st.line_chart({"accuracy": controller.accuracy}, x="accuracy")


    train_button = st.button("start", use_container_width=True)
    if train_button:
        with st.spinner("model is training......"):
            train_button = False
            run()
        st.success('finish!', icon="✅")


