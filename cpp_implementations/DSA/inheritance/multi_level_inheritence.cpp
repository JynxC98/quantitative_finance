/**
 * @brief This script is used to demonstrate the example usecase of multi-level
 * inheritance.
 */

#include <iostream>
#include <string>

class Animal
{
private:
    int num_legs;
    bool isCarnivore;
    std::string environment;

public:
    std::string name;

    // Initializing a constructor
    Animal(std::string n, int legs, bool carnivore, std::string env)
        : name(n), num_legs(legs), isCarnivore(carnivore), environment(env) {}

protected:
    int getNumLegs() const { return num_legs; }
    bool getIsCarnivore() const { return isCarnivore; }
    std::string getEnvironment() const { return environment; }

public:
    // Public methods available to all
    virtual void makeSound() const
    {
        std::cout << name << " makes a generic animal sound" << std::endl;
    }

    void displayInfo() const
    {
        std::cout << "Name: " << name << std::endl;
        std::cout << "Legs: " << num_legs << std::endl;
        std::cout << "Carnivore: " << (isCarnivore ? "Yes" : "No") << std::endl;
        std::cout << "Environment: " << environment << std::endl;
    }

    virtual ~Animal() {} // Virtual destructor for proper cleanup
};

// Mammal inherits from Animal
class Mammal : public Animal
{
protected:
    bool hasFur;
    int gestationPeriod; // in days

public:
    Mammal(std::string n, int legs, bool carnivore, std::string env, bool fur, int gestation)
        : Animal(n, legs, carnivore, env), hasFur(fur), gestationPeriod(gestation) {}

    void nurseYoung() const
    {
        std::cout << name << " is nursing its young!" << std::endl;
    }

    void makeSound() const override
    {
        std::cout << name << " makes a mammal sound" << std::endl;
    }
};

// Whale inherits from Mammal (and indirectly from Animal)
class Whale : public Mammal
{
private:
    double length; // in meters

public:
    Whale(std::string n, double len, int gestation = 365)
        : Mammal(n, 0, true, "Ocean", true, gestation), length(len) {}

    void makeSound() const override
    {
        std::cout << name << " the whale sings!" << std::endl;
    }

    void swim() const
    {
        std::cout << name << " is swimming in the ocean!" << std::endl;
    }

    void blowHole() const
    {
        std::cout << name << " blows water from its blowhole!" << std::endl;
    }
};