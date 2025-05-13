import { useEffect, useState } from "react";
import backendClient from "@/backendClient";
import BackdropWithSpinner from "@/components/ui/backdropwithspinner";
import { Card } from "@/components/ui/card";
import useGlobalStore from "@/store/store";

const Examples = () => {
  const [isLoading, setLoading] = useState(false);
  const [users, setUsers] = useState([]);

  const fetchUsers = async() => {
    setLoading(true);
    const response = await backendClient.get("/users");
    if(response.data)
      setUsers(response.data)
    setLoading(false);
  }
  const error = useGlobalStore(state => state.error);

  useEffect(() => {
    fetchUsers();
  }, []);

  return (
    <>
      <h1>Users</h1>
      {!error && users.map(user => <Card>{user.name}</Card>)}
      {isLoading && <BackdropWithSpinner />}
    </>
  );
}

export default Examples;