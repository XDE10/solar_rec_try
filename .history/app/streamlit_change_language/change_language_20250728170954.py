import os
from .config import Config
from .replace import Replace
from typing import Optional, Tuple, List, Dict


class ChangeStLanguage:
    def __init__(self):
        # 判断有没有安装streamlit
        streamlit_path, streamlit_version = self.get_streamlit()
        if not streamlit_path:
            raise ImportError("streamlit module not found")
        else:
            self.streamlit_version = streamlit_version
            self.streamlit_path = streamlit_path.replace("__init__.py", "")
            self.js_path = os.path.join(self.streamlit_path, "static\static\js")
            self.static_path = os.path.join(self.streamlit_path, "static")

            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            self.config_path = os.path.join(BASE_DIR, "config.ini")
            self.config = Config(self.config_path)
            # 将基本信息写入配置文件
            self.write_message()

    def get_streamlit(self) -> Optional[Tuple[Optional[str], str]]:
        """
        用途：获取streamlit模块的信息
        返回：None或模块地址,版本
        """
        try:
            import streamlit
        except Exception:
            return
        else:
            return streamlit.__file__, streamlit.__version__

    def write_message(self):
        """
        用途：初始化时读取streamlit各组件对应的js文件名称，并写入配置文件中
        """
        config = self.config.read()
        sections = config.sections()
        config.set('streamlit', 'version', self.streamlit_version)
        if "js_path" not in sections:
            config.add_section("js_path")
        if "py_path" not in sections:
            config.add_section("py_path")
        js_dict = self.load_js_path()
        for module, js_path in js_dict.items():
            config.set('js_path', module, js_path)
        multiselect_py_path = os.path.join(self.streamlit_path, "elements\widgets\multiselect.py")
        config.set('py_path', "multiselect", multiselect_py_path)
        self.config.save(config)

    def get_js_files_list(self, path: str) -> List[str]:
        """
        用途：读取js目录下的每个js文件
        返回：str列表
        """
        files_list=[]
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.js'):
                    files_list.append(os.path.join(root, file))
        return files_list

    def check_mes_in_file(self, file_path: str, mes: str) -> bool:
        """
        用途：检测文本是否存在于文件中
        返回：bool
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            return mes in content

    def load_js_path(self) -> Dict[str, str]:
        """
        用途：检测组件值存在哪个js中
        返回：[{组件名:js路径}]
        """
        config = self.config.read()
        module_items = config.items("module")
        js_files_list = self.get_js_files_list(self.js_path)
        module_dict = {}
        for file_path in js_files_list:
            for module, value in module_items:
                if self.check_mes_in_file(file_path, value):
                    module_dict.update({module: file_path})
        return module_dict

    def get_now_config(self) -> tuple[str, str]:
        """
        用途：获取配置文件中的语言及版本
        返回：语言，版本
        """
        config = self.config.read()
        language = config.get("streamlit", "language")
        version = config.get("streamlit", "version")
        return language, version

    def update_config(self, language: str):
        if language not in ["en", "cn"]:
            raise ValueError("language must be 'en' or 'cn'")
        else:
            config = self.config.read()
            config.set('streamlit', 'language', language)
            self.config.save(config)

    def change_text(self, file_path:str, replace_value:list[dict[str:str]], language:str):
        with open(file_path, 'r+', encoding='utf-8') as file:
            content = file.read()
            for i in replace_value:
                if language == "cn":
                    for k, v in i.items():
                        content = content.replace(k, v)
                if language == 'en':
                    for k, v in i.items():
                        content = content.replace(v, k)
            file.seek(0)
            file.truncate()
            file.writelines(content)
    def replace_text(self, language:str):
        config = self.config.read()
        js_items = config.items("js_path")
        py_items = config.items("py_path")
        for js_module, js_path in js_items:
            self.change_text(js_path, Replace.get_js_value(js_module), language)
        for py_module, py_path in py_items:
            self.change_text(py_path, Replace.get_py_value(py_module), language)

    def change(self, language: str):
        config_language, cofig_verrsion = self.get_now_config()
        _, now_version = self.get_streamlit()
        if language not in ["en", "cn"]:
            raise ValueError("language must be 'en' or 'cn'")
        else:
            if now_version != cofig_verrsion:
                self.write_message()
            if language != config_language:
                self.replace_text(language)
                self.update_config(language)

changeStLanguage = ChangeStLanguage()