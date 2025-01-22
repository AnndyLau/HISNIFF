import json
import requests
import random
#random.seed(123)
class AccessToken(object):
    # 微信公众测试号账号（填写自己的）
    APPID = "wx137a520102c0fd0d"
    # 微信公众测试号密钥（填写自己的）
    APPSECRET = "1c770b7948afaa85fe52a46757c47541"

    def __init__(self, app_id=APPID, app_secret=APPSECRET) -> None:
        self.app_id = app_id
        self.app_secret = app_secret

    def get_access_token(self) -> str:
        """
        获取access_token凭证
        :return: access_token
        """
        url = f"https://api.weixin.qq.com/cgi-bin/token?grant_type=client_credential&appid" \
              f"={self.app_id}&secret={self.app_secret}"
        resp = requests.get(url)
        result = resp.json()
        if 'access_token' in result:
            return result["access_token"]
        else:
            print(result)


class SendMessage(object):
    # 消息接收者
    TOUSER = 'oLWNj6ZZ3gH3aMV18-J1qudg0FYg'
    # 消息模板id
    TEMPLATE_ID = 'c19Cpr97lj47xdYGsmbfvOCnKqQOpn5rKfPf5_5BwwA'
    # 点击跳转链接（可无）
    CLICK_URL = ' '

    def __init__(self, touser=TOUSER, template_id=TEMPLATE_ID, click_url=CLICK_URL) -> None:
        """
        构造函数
        :param touser: 消息接收者
        :param template_id: 消息模板id
        :param click_url: 点击跳转链接（可无）
        """
        self.access_token = AccessToken().get_access_token()
        self.touser = touser
        self.template_id = template_id
        self.click_url = click_url

    def random_color(self):
        colors1 = '0123456789ABCDEF'
        num = "#"
        for i in range(6):
            num += random.choice(colors1)
        return num
    def get_send_data(self, json_data) -> object:
        """
        获取发送消息data
        :param json_data: json数据对应模板
        :return: 发送的消息体
        """
        return {
            "touser": self.touser,
            "template_id": self.template_id,
            "url": self.click_url,
            "topcolor": "#FF0000",
            # json数据对应模板
            "data": {
                "shu": {"value": json_data["shu"], "color": self.random_color()},
                "train_acc": {"value": json_data["train_acc"], "color": self.random_color()},
                "test_acc": {"value": json_data["test_acc"], "color": self.random_color()},
                "epoch": {"value": json_data["epoch"], "color": self.random_color()},
                "second": {"value": json_data["second"], "color": self.random_color()},
            }
        }

    def send_message(self, json_data) -> None:
        """
        发送消息
        :param json_data: json数据
        :return:
        """
        # 模板消息请求地址
        url = f"https://api.weixin.qq.com/cgi-bin/message/template/send?access_token={self.access_token}"
        data = json.dumps(self.get_send_data(json_data))
        resp = requests.post(url, data=data)
        result = resp.json()
        # 有关响应结果，我有整理成xml文档（官方全1833条），免费下载：https://download.csdn.net/download/sxdgy_/86263090
        if result["errcode"] == 0:
            print("Message sent successfully!")
        else:
            print(result)


# if __name__ == '__main__':
#     # 实例SendMessage
#     sm = SendMessage()
#     name = "function test"
#     loss = 0.12
#     finalAcc = 92.32
#     bestEpoch = 132
#     bestAcc = 93.12
#     json_data = {
#         "shu": name,
#         "train_acc": loss,
#         "test_acc": finalAcc,
#         "epoch": bestEpoch,
#         "second": bestAcc
#     }
#     # 发送消息
#     sm.send_message(json_data=json_data)
