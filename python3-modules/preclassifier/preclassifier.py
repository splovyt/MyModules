class Preclassifier:

    def __init__(self, X, y):
        assert type(X) == list, "X should be a list to ensure proper function."
        assert type(y) == list, "y should be a list to ensure proper function."

        # the lookup table
        self.lookup_table = dict()

        # add to the lookup table
        for x, y in zip(X, y):
            # generate the str representation of x to allow use as a key
            x_key = repr(x)

            try:
                # add y to the corresponding x key
                self.lookup_table[x_key].append(y)
            except KeyError:
                # or to a new x key
                self.lookup_table[x_key] = [y]

        # sort the y's per x by highest occurence
        for x_key, y_list in self.lookup_table.items():
            # sort the counted y values by decreasing amount
            self.lookup_table[x_key] = sorted([(z, y_list.count(z)) for z in set(y_list)],
                                              key=lambda x: x[1], reverse=True)


    def add_single_entry(self, X_entry, y_entry):
        # check if the entry is already in the lookup table
        try:
            # the sorted y list
            y_list_old = self.lookup_table[repr(X_entry)]

            # update the list with new y and therefore new counts
            y_list_new = y_list_old.copy()
            in_list = False
            for idx, (y, count) in enumerate(y_list_old):
                try:
                    if y == y_entry:
                        # if found, increase the count by one
                        y_list_new[idx] = (y, y_list_old[idx][1] + 1)
                        in_list = True
                        break
                except:
                    if repr(y) == repr(y_entry):
                        # if found, increase the count by one
                        y_list_new[idx] = (y, y_list_old[idx][1] + 1)
                        in_list = True
                        break

            if not in_list:
                # if the item was not found in the list, add to the end with count one
                y_list_new.append((y_entry, 1))

            # now update the lookup table with the new and sorted list
            self.lookup_table[repr(X_entry)] = sorted(y_list_new, key=lambda x: x[1], reverse=True)

        except KeyError:
            # if the entry is not yet in the lookup table, create the entry
            self.lookup_table[repr(X_entry)] = [(y_entry, 1)]

    def add_multiple_entries(self, X_entries, y_entries):
        assert type(X_entries) == list, "X_entries should be a list to ensure proper function."
        assert type(y_entries) == list, "y_entries should be a list to ensure proper function."

        for x, y in zip(X_entries, y_entries):
            self.add_single_entry(x, y)

    def single_query(self, X_entry):
        # return the highest occurring y
        try:
            return self.lookup_table[repr(X_entry)][0][0]
        except KeyError:
            return None

    def multiple_query(self, X_entries):
        # perform a single query multiple times
        assert type(X_entries) == list, "X_entries should be a list to ensure proper function."
        return [self.single_query(x) for x in X_entries]

