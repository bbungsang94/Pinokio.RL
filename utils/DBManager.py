import sys
import platform
import mariadb


def load_default():
    default = dict()
    default['host'] = '127.0.0.1'
    default['user'] = 'root'
    default['password'] = '151212kyhASH@'
    default['port'] = int(3306)
    default['database'] = 'lv2meblayer4'

    return default


class MariaManager:
    def __init__(self, config):
        self.DML = None
        self.DDL = None

        checklist = ['host', 'user', 'password', 'port', 'database']
        default = load_default()
        for checker in checklist:
            if checker in config:
                default[checker] = config[checker]
        self.__connect(default)

    def release(self):
        self.Connector.close()

    def __connect(self, config):
        try:
            if platform.system() == 'Windows':
                auto = True
            else:  # Mac에서 트랜잭션 작업을 진행함
                auto = True

            self.Connector = mariadb.connect(user=config['user'], password=config['password'], host=config['host'],
                                             port=config['port'], database=config['database'], autocommit=auto)
            self.Cursor = self.Connector.cursor()
            self.DML = self.__DataManipulationLanguage(self.Connector)
            self.DDL = self.__DataDefinitionLanguage(self.Connector)
        except mariadb.Error as e:
            print(f"Maria DB connecting error: {e}")
            sys.exit(1)

    class __DataManipulationLanguage:
        def __init__(self, connector):
            self.Connector = connector
            self.Cursor = self.Connector.cursor()

        # region Select module
        def select(self, tb, col, cond=None):
            query = 'SELECT ' + col + ' FROM ' + tb

            if cond is not None:
                query += ' WHERE ' + cond
            self.Cursor.execute(query)
            return self.Cursor.fetchall()

        def select_all(self, tb, cond=None):
            query = 'SELECT * FROM ' + tb

            if cond is not None:
                query += ' WHERE ' + cond

            self.Cursor.execute(query)
            return self.Cursor.fetchall()

        def distinct(self, tb, col, cond=None):
            query = 'SELECT DISTINCT ' + col + ' FROM ' + tb

            if cond is not None:
                query += ' WHERE ' + cond

            self.Cursor.execute(query)
            return self.Cursor.fetchall()

        # endregion
        # region Insert module
        def insert(self, tb, values):
            rtn = False
            columns = self.get_columns(tb=tb)

            for index, column in enumerate(columns):
                columns[index] = '`' + column[0] + '`'

            for index, value in enumerate(values):
                if value is None:
                    del values[index]
                    del columns[index]

            if len(columns) == len(values):
                query = 'INSERT INTO ' + tb + ' ('
                query += ", ".join(columns) + ') VALUES ('
                query += ", ".join(values) + ')'
                self.Cursor.execute(query)
                self.Connector.commit()
                rtn = "Last Inserted ID: " + str(self.Cursor.lastrowid)

            return rtn

        # endregion
        # region Delete module
        def delete(self, tb, cond):
            query = 'DELETE FROM ' + tb + ' WHERE ' + cond
            self.Cursor.execute(query)
            self.Connector.commit()
            return "Delete condition: " + str(cond)

        # endregion
        # region Update module
        def update(self, tb, columns, values, cond=None):
            query = 'UPDATE ' + tb + ' SET '
            for index, column in enumerate(columns):
                if index != 0:
                    query += ", "
                query += column + ' = ' + values[index]

            if cond is not None:
                query += ' WHERE ' + cond
            self.Cursor.execute(query)
            #self.Connector.commit()
            return "Update columns: " + ", ".join(columns)

        # endregion
        # region Custom method
        def manual_select(self, query):
            self.Cursor.execute(query)
            return self.Cursor.fetchall()

        def manual_insert(self, query):
            self.Cursor.execute(query)
            self.Connector.commit()
            return "Last Inserted ID: " + str(self.Cursor.lastrowid)

        def get_columns(self, tb, on_tuple=True):
            columns = self.select(col='column_name', tb='information_schema.columns',
                                  cond='table_name="' + tb + '" AND table_schema = "' + self.Connector.database + '";')
            if on_tuple is False:
                for index, value in enumerate(columns):
                    columns[index] = value[0]
            return columns
        # endregion

    class __DataDefinitionLanguage:
        def __init__(self, connector):
            self.Connector = connector
            self.Cursor = self.Connector.cursor()

        # region Create module
        def create(self, tb, columns):
            count = 0
            query = 'CREATE TABLE ' + tb + '( '
            for name, attributes in columns:
                query += name + ' ' + attributes
                if len(columns) != count:
                    query += ', '
            query += ' );'
            self.Cursor.execute(query)
            self.Connector.commit()
            return "Last Created table name: " + tb

        def manual_create(self, query):
            self.Cursor.execute(query)
            self.Connector.commit()

        # endregion
        # region Drop module
        def drop(self, tb, exist=False):
            if exist:
                query = "DROP TABLE IF EXISTS " + tb
            else:
                query = "DROP TABLE " + tb
            self.Cursor.execute(query)
            self.Connector.commit()

        # endregion
        # region Truncate module
        def truncate(self, tb):
            query = "TRUNCATE TABLE " + tb
            self.Cursor.execute(query)
            self.Connector.commit()
        # endregion


if __name__ == '__main__':
    manager = MariaManager(load_default())
    val = manager.DML.select_all(tb='mcs_order')
    val = ['777', '"TESTORDER777"', '"2021-10-25 18:07:11"', '50', '50',
           '"813"', '"RP4_1"', '"404"', '"SLT3_1"', '"EBEDFY0159"', '"None"',
           '0', '"Registered"', '0', '0', '0', '0',
           '"None"', '"None"', '"None"', '"None"', None, '0', None]
    val = manager.DML.insert(tb='mcs_order', values=val)
    # manager.DML.delete(tb='mcs_order', cond='uid=777')
    val = manager.DML.update(tb='mcs_order', columns=["`uid`", "`priority`", "`from_node_mcs`"],
                             values=['482', '49', '"RP7_1"'], cond='`order_id` = "TESTORDER777"')

    print(val, '\n')
    # manager.__del__()
