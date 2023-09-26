import time
import pandas as pd
import streamlit as st

class CheckBoxSample01(object):
    def __init__(self):
        # 一覧のチェックボックスの状態を
        # key=login_user_id/value=True/False で保持する
        self.selected_checkbox={}

    # チェックボックスのチェックON状態の key 一覧を返します
    def list_checked_ids(self):
        list_id = []
        for key_id, selected in self.selected_checkbox.items():
            if selected:
                list_id.append(key_id)
        return list_id

    # フォーム表示
    def show(self, df):
        st.header("チェックボックス状態確認 実装デモ")
        # テーブル表示
        col_size = [1, 10]
        columns = st.columns(col_size)
        headers = ["check","メールアドレス"]
        for col, field_name in zip(columns, headers):
            col.write(field_name)

        for index, item in df.iterrows():
            (
                button_col,
                user_email
            ) = st.columns(col_size)
            with button_col:
                is_selected = st.checkbox("", value=item.user_email, key="check_{}".format(user_email))
                self.selected_checkbox[item.login_user_id] = is_selected
            user_email.write(item.user_email)

        st.button(label="保存", on_click=self.onclick_update, args=())
        self.messageHolder = st.empty() # for error message


    # ボタン押下時
    def onclick_update(self):
        # チェックON状態の login_user_id 一覧の取得
        list_login_user_id = self.list_checked_ids()

        # 選択状態チェック
        if len(list_login_user_id) == 0:
            self.show_error_message("チェックなしです")
            return
        else:
            self.show_error_message("チェックあり")


    # エラーメッセージ表示
    def show_error_message(self, msg):
        self.messageHolder.write(msg)
        time.sleep(1)

def main():
    page = CheckBoxSample01()
    df1 = pd.DataFrame(
      data={
        'login_user_id':[ 1, 3, 6 ],
        'user_email':["aaa@test.com","bbb@test.com","cccc@test.com"]
      }
    )
    page.show(df1)


if __name__ == '__main__':
    main()
