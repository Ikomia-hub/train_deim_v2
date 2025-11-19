from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def get_process_factory(self):
        # Instantiate algorithm object
        from train_deim_v2.train_deim_v2_process import TrainDeimV2Factory
        return TrainDeimV2Factory()

    def get_widget_factory(self):
        # Instantiate associated widget object
        from train_deim_v2.train_deim_v2_widget import TrainDeimV2WidgetFactory
        return TrainDeimV2WidgetFactory()
